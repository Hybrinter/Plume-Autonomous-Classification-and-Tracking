"""General env-driven validation-harness API: build + step a flight system for any profile.

This is the composition-root surface the GSE in-process backend drives. GSE imports only
flight.libs and sim, so sim.sil exposes a general validation harness here rather than letting
GSE touch flight.core/flight.payload/flight.fault directly. build_validation_system honors the
full PactConfig.environment axis vector via flight.core.select_drivers -- so for a sil-link-real
profile (link="real") it yields a RealStationLink while every other axis stays a sim stand-in.

Unlike build_sil_system (which forces an all-"sim" environment for the deterministic SIL), this
builder is env-driven: it passes config.environment through untouched. The returned
ValidationSystem is Protocol-typed (HAL Protocols only), so it carries whatever concrete drivers
the axes selected without the holder ever naming a concrete driver.

ValidationHarness is the general single-threaded stepper: it reuses sim.sil.stepping.step_once
(the one source of truth for a cycle) and threads the payload ControlState + FDIR watchdog entries
in and out, exactly as SilHarness does, but over the Protocol-typed ValidationSystem.

Contains:
  - ValidationSystem: Protocol-typed holder of the wired apps + bus/clock + selected drivers.
  - build_validation_system: env-driven builder (select_drivers -> build_apps) for any profile.
  - ValidationHarness: deterministic single-threaded stepper (step / run_steps).
  - load_profile_config: load default.toml + a profile override into a PactConfig (raises on Err).

Satisfies: REQ-OPER-HIGH-002.
"""

from __future__ import annotations

# stdlib
import tempfile
from dataclasses import dataclass, replace

# internal
from flight.core.composition import MONITORED_SUBSYSTEMS, SystemApps, build_apps
from flight.core.config_loader import load_config
from flight.core.select_drivers import SimDriverInputs, select_drivers
from flight.fault.watchdog import WatchdogEntry
from flight.hal.interfaces import GimbalActuator, ImagingSensor, ScalarSensor, StationLink
from flight.libs.bus import MessageBus
from flight.libs.config import PactConfig
from flight.libs.time import ManualClock
from flight.libs.types import GimbalState, Ok
from flight.payload.calibration_io import build_identity_calibration
from flight.payload.control import ControlState

from sim.sil.stepping import step_once


@dataclass(frozen=True)
class ValidationSystem:
    """The wired flight system: apps + shared bus/clock + the env-selected HAL drivers.

    Protocol-typed throughout (flight.hal.interfaces), so it carries whatever concrete
    drivers the PactConfig.environment axes selected -- a SimSensor or a RealSensor, a
    SimStationLink or a RealStationLink -- without the holder naming a concrete type. This
    is what lets the GSE in-process backend drive any profile through sim only. frozen=True
    without slots is intentional: a holder of Protocol-typed fields does not need slots.
    """

    apps: SystemApps
    bus: MessageBus
    clock: ManualClock
    sensor: ImagingSensor
    gimbal: GimbalActuator
    station: StationLink
    thermal_sensor: ScalarSensor
    power_sensor: ScalarSensor


def build_validation_system(
    config: PactConfig,
    clock: ManualClock,
    sim_inputs: SimDriverInputs | None = None,
    uplink_key: bytes = b"sil-test-key-0000000000000000000",
) -> ValidationSystem:
    """Wire the flight apps over the env-selected drivers on a fresh bus, for any profile.

    Constructs a fresh MessageBus, resolves config.environment to a concrete Drivers bundle
    via flight.core.select_drivers (the one env-driven selection path the flight entry uses),
    builds an identity MosaicCalibration sized to the sensor config, and wires every app via
    the driver-agnostic build_apps. Unlike build_sil_system, the environment axis vector is
    passed through untouched, so a 'real' axis yields the real driver (e.g. link="real" ->
    RealStationLink).

    Args:
        config: The validated PactConfig; its environment axes drive driver selection.
        clock: The ManualClock shared by all apps (timestamps; the harness advances `now`).
        sim_inputs: The sim construction inputs (frames, detector, packets, readings);
            required by select_drivers when any selected axis is 'sim'.
        uplink_key: The HMAC-SHA256 secret the iss_iface app uses to authenticate inbound
            TC packets. Defaults to a fixed test key; pass explicitly in command-path tests.

    Returns:
        A ValidationSystem holding the wired apps, the shared bus/clock, and the
        Protocol-typed drivers select_drivers resolved from the environment axes.

    Notes:
        The returned drivers fields are the exact Drivers.* objects select_drivers built, so
        the holder needs no cast: the Drivers fields are already declared with the same HAL
        Protocols ValidationSystem uses.

        Storage is redirected to a fresh temp directory so the deterministic in-process harness
        is hermetic (no repo pollution) and isolated per build; the flight entry keeps the
        configured data_root.
    """
    config = replace(
        config,
        storage=replace(config.storage, data_root=tempfile.mkdtemp(prefix="pact-sil-storage-")),
    )
    bus = MessageBus()
    drivers = select_drivers(config, clock, sim_inputs)
    calib = build_identity_calibration(config.sensor.height_px, config.sensor.width_px)
    apps = build_apps(config, bus, clock, drivers, MONITORED_SUBSYSTEMS, calib, uplink_key)
    return ValidationSystem(
        apps=apps,
        bus=bus,
        clock=clock,
        sensor=drivers.sensor,
        gimbal=drivers.gimbal,
        station=drivers.station,
        thermal_sensor=drivers.thermal_sensor,
        power_sensor=drivers.power_sensor,
    )


class ValidationHarness:
    """Deterministic single-threaded driver for a ValidationSystem (no scheduler threads)."""

    def __init__(self, system: ValidationSystem) -> None:
        """Seed the payload control state and the FDIR watchdog entries.

        Args:
            system: The wired ValidationSystem to drive.
        """
        self._system = system
        self._payload_state: ControlState = system.apps.payload.controller.initial_state()
        self._fault_entries: dict[str, WatchdogEntry] = system.apps.fault.initial_entries()

    def payload_gimbal_state(self) -> GimbalState:
        """Return the payload arbiter's current GimbalState (test/inspection accessor)."""
        return self._payload_state.arbiter.gimbal_state

    def step(self, now: float) -> None:
        """Advance every subsystem one cycle over the shared bus (delegates to step_once).

        Args:
            now: Monotonic seconds for the arbiter and watchdog (advanced by the caller).
        """
        system = self._system
        self._payload_state, self._fault_entries = step_once(
            system.apps,
            system.sensor,
            system.gimbal,
            system.bus,
            system.clock,
            now,
            self._payload_state,
            self._fault_entries,
        )

    def run_steps(self, count: int, dt: float = 1.0) -> None:
        """Run count deterministic steps, advancing `now` and the shared clock by dt each step.

        Advancing the shared ManualClock each step lets time-integrating sim drivers (e.g. the
        SimGimbal first-order dynamics) integrate between steps; for real drivers the advanced
        `now` still drives the arbiter and watchdog deterministically.

        Args:
            count: Number of steps to run.
            dt: Seconds to advance `now` per step.
        """
        now = 0.0
        for _ in range(count):
            now += dt
            self._system.clock.advance(dt)
            self.step(now)


def load_profile_config(config_path: str, override_path: str) -> PactConfig:
    """Load config_path merged with a profile override into a PactConfig, raising on failure.

    A composition-root convenience: a config load failure at startup is unrecoverable, so this
    raises rather than returning a Result (per the Result-vs-startup-exception distinction).

    Args:
        config_path: Path to the base TOML config (typically "config/default.toml").
        override_path: Path to the deployment-profile override TOML (e.g. "profiles/sil.toml").

    Returns:
        The merged, validated PactConfig.

    Raises:
        ValueError: If load_config returns an Err (missing file, parse, or validation error).
    """
    result = load_config(config_path, override_path)
    if not isinstance(result, Ok):
        raise ValueError(f"config load failed: {result.error}")
    return result.value
