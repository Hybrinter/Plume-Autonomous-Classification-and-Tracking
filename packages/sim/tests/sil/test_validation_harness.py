"""Env-driven validation system builder + harness: GSE drives flight through sim only."""

import dataclasses

import pytest
from flight.core.select_drivers import SimDriverInputs
from flight.hal.drivers_sim import SimSensor, SimStationLink
from flight.libs.config import EnvironmentConfig, PactConfig
from flight.libs.messages import InferenceResultMsg
from flight.libs.time import ManualClock
from sim.scene import build_frames, plume_detector
from sim.sil import (
    ValidationHarness,
    ValidationSystem,
    build_validation_system,
    load_profile_config,
)


def _all_sim_config() -> PactConfig:
    """Return a PactConfig whose every deployment axis is a sim stand-in."""
    sim_env = EnvironmentConfig(
        sensor="sim",
        gimbal="sim",
        compute="sim",
        link="sim",
        clock="sim",
        host="x86_64",
    )
    return dataclasses.replace(PactConfig(), environment=sim_env)


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
    """Four steps over four frames drive exactly four InferenceResultMsg publications."""
    system = build_validation_system(_all_sim_config(), ManualClock(), _sim_inputs())
    inf_sub = system.bus.subscribe(InferenceResultMsg)

    ValidationHarness(system).run_steps(4)

    inference_count = 0
    while not inf_sub.empty():
        inf_sub.get_nowait()
        inference_count += 1
    assert inference_count == 4


def test_load_profile_config_loads_sim_profile() -> None:
    """The SIL profile override yields an all-sim environment PactConfig."""
    config = load_profile_config("config/default.toml", "profiles/sil.toml")

    env = config.environment
    assert (env.sensor, env.gimbal, env.compute, env.link, env.clock) == (
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
