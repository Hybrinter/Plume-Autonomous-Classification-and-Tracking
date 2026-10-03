"""Env-driven validation system builder + harness: GSE drives flight through sim only."""

import dataclasses
import threading

import pytest
from flight.core.select_drivers import SimDriverInputs
from flight.hal.drivers_sim import SimSensor, SimStationLink
from flight.libs.config import DriverConfig, PactConfig
from flight.libs.messages import InferenceResultMsg
from flight.libs.time import ManualClock
from flight.libs.types import FaultCode, Ok, Result, SystemMode
from flight.payload.graphs.base import InitVerificationResult, InitVerificationStatus
from flight.payload.lifecycle import (
    LifecycleObservation,
    PendingInitializationVerifier,
)
from sim.scene import build_frames, plume_detector
from sim.sil import (
    ValidationHarness,
    ValidationSystem,
    build_validation_system,
    load_profile_config,
    publish_activation,
)


class _RecordingVerifier:
    """Deterministic INIT verifier recording calls and staying pending."""

    def __init__(self) -> None:
        self.calls = 0

    def verify(
        self, observation: LifecycleObservation, cancel: threading.Event
    ) -> Result[InitVerificationResult, FaultCode]:
        """Record the call and answer PENDING on the observed key."""
        del cancel
        self.calls += 1
        return Ok(
            InitVerificationResult(
                activation_key=observation.inputs.activation_key,
                status=InitVerificationStatus.PENDING,
            )
        )


def _all_sim_config() -> PactConfig:
    """Return a PactConfig whose every deployment axis is a sim stand-in."""
    sim_drivers = DriverConfig(
        sensor="sim",
        gimbal="sim",
        compute="sim",
        link="sim",
        clock="sim",
        host="x86_64",
    )
    base = PactConfig()
    return dataclasses.replace(
        base,
        drivers=sim_drivers,
        gimbal=dataclasses.replace(
            base.gimbal,
            simulation=dataclasses.replace(base.gimbal.simulation, encoder_noise_deg=0.0),
        ),
    )


def _sim_inputs() -> SimDriverInputs:
    """Return deterministic sim driver inputs: plume frames + scripted detector.

    Thermal/power get one nominal reading each (the SimScalarSensor holds its final value
    once exhausted, so a single reading suffices for any step count).
    """
    return SimDriverInputs(
        frames=build_frames(4, seed=0),
        detector=plume_detector(),
        inbound_packets=[],
        thermal_readings=[25.0],
        power_readings=[30.0],
    )


def test_build_validation_system_yields_sim_drivers() -> None:
    """An all-sim config builds a ValidationSystem backed by SimSensor + SimStationLink."""
    system = build_validation_system(_all_sim_config(), ManualClock(), _sim_inputs())

    assert isinstance(system, ValidationSystem)
    assert isinstance(system.sensor, SimSensor)
    assert isinstance(system.station, SimStationLink)


def test_validation_harness_drives_inference_per_frame() -> None:
    """Four steps at duty 0.5 publish floor(4 * 0.5) inference results."""
    system = build_validation_system(_all_sim_config(), ManualClock(), _sim_inputs())
    publish_activation(system, SystemMode.OPERATE, sequence=1)
    inf_sub = system.bus.subscribe(InferenceResultMsg)

    ValidationHarness(system).run_steps(4)

    inference_count = 0
    while not inf_sub.empty():
        inf_sub.get_nowait()
        inference_count += 1
    assert inference_count == 2


def test_load_profile_config_loads_sim_profile() -> None:
    """The SIL profile override yields an all-sim driver PactConfig."""
    config = load_profile_config("config/default.toml", "profiles/sil.toml")

    drivers = config.drivers
    assert (drivers.sensor, drivers.gimbal, drivers.compute, drivers.link, drivers.clock) == (
        "sim",
        "sim",
        "sim",
        "sim",
        "sim",
    )


def test_load_profile_config_bad_override_raises() -> None:
    """A nonexistent override path surfaces as a ValueError startup failure."""
    with pytest.raises(ValueError):
        load_profile_config("config/default.toml", "profiles/does-not-exist.toml")


def test_validation_system_forwards_initialization_verifier() -> None:
    """An explicit verifier reaches the payload executor through the builder."""
    verifier = _RecordingVerifier()
    system = build_validation_system(
        _all_sim_config(),
        ManualClock(),
        _sim_inputs(),
        initialization_verifier=verifier,
    )

    assert system.apps.payload.lifecycle._verifier is verifier


def test_validation_system_default_verifier_stays_pending() -> None:
    """All-sim composition never implies a passing verifier: default is pending."""
    system = build_validation_system(_all_sim_config(), ManualClock(), _sim_inputs())

    assert isinstance(system.apps.payload.lifecycle._verifier, PendingInitializationVerifier)
