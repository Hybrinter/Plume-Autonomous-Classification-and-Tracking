"""SIL runner: build the flight apps over sim drivers and step them deterministically.

build_sil_system constructs the sim drivers + scripted detector, bundles them as
Drivers, and calls the Phase-9 driver-agnostic build_apps -- so the SIL exercises the
exact same wiring the flight entry uses. SilHarness drives the apps single-threaded:
each step acquires + processes one frame, samples housekeeping, pumps the ISS bridge,
publishes per-subsystem liveness heartbeats, then runs the FDIR tick -- all over the
shared in-process bus, with `now` advanced explicitly for full determinism.

Contains:
  - SilSystem: the wired apps + bus + clock + the concrete sim drivers (for inspection).
  - build_sil_system: construct the sim drivers and wire the apps via build_apps.
  - SilHarness: deterministic single-threaded stepper (step / run_steps).
"""

from __future__ import annotations

# stdlib
import dataclasses
from dataclasses import dataclass
from typing import cast

# internal
from flight.core.composition import MONITORED_SUBSYSTEMS, SystemApps, build_apps
from flight.core.select_drivers import SimDriverInputs, select_drivers
from flight.fault.watchdog import WatchdogEntry
from flight.hal.drivers_sim import SimGimbal, SimScalarSensor, SimSensor, SimStationLink
from flight.libs.bus import MessageBus
from flight.libs.config import EnvironmentConfig, PactConfig
from flight.libs.time import ManualClock
from flight.libs.types import GimbalState, MosaicFrame
from flight.payload.calibration_io import build_identity_calibration
from flight.payload.control import ControlState
from flight.payload.model import ScriptedDetector

from sim.sil.stepping import step_once


@dataclass(frozen=True)
class SilSystem:
    """The wired SIL system: apps + shared bus/clock + the concrete sim drivers."""

    apps: SystemApps
    bus: MessageBus
    clock: ManualClock
    sensor: SimSensor
    gimbal: SimGimbal
    station: SimStationLink
    thermal_sensor: SimScalarSensor
    power_sensor: SimScalarSensor


def build_sil_system(
    config: PactConfig,
    clock: ManualClock,
    frames: list[MosaicFrame],
    detector: ScriptedDetector,
    inbound_packets: list[bytes] | None = None,
    thermal_readings: list[float] | None = None,
    power_readings: list[float] | None = None,
    uplink_key: bytes = b"sil-test-key-0000000000000000000",
) -> SilSystem:
    """Construct the sim drivers and wire the flight apps over a fresh bus via build_apps.

    Args:
        config: The PactConfig to wire the apps with.
        clock: The ManualClock shared by all apps (timestamps; the harness advances `now`).
        frames: Raw mosaic frames the SimSensor replays.
        detector: The ScriptedDetector backing the payload.
        inbound_packets: CCSDS TC packets the SimStationLink delivers via the ISS bridge.
        thermal_readings: Temperature readings the thermal sensor replays (Celsius).
        power_readings: Power readings the electrical sensor replays (Watts).
        uplink_key: The HMAC-SHA256 secret used by the iss_iface app to authenticate
            inbound TC packets. Defaults to a fixed SIL test key; pass explicitly in
            command-path SIL tests that build packets with build_tc_packet.

    Returns:
        A SilSystem holding the wired apps, the shared bus/clock, and the sim drivers.

    Notes:
        Delegates concrete-driver construction to flight.core.select_drivers with an
        all-"sim" EnvironmentConfig (host "x86_64"), so the SIL exercises the exact
        same selection path the flight entry uses. The returned SilSystem casts the
        Protocol-typed Drivers fields back to their concrete sim types for inspection.
    """
    bus = MessageBus()
    sim_inputs = SimDriverInputs(
        frames=frames,
        detector=detector,
        inbound_packets=inbound_packets or [],
        thermal_readings=thermal_readings or [],
        power_readings=power_readings or [],
    )
    sil_env = EnvironmentConfig(
        sensor="sim",
        gimbal="sim",
        compute="sim",
        link="sim",
        clock="sim",
        host="x86_64",
    )
    sil_config = dataclasses.replace(config, environment=sil_env)
    drivers = select_drivers(sil_config, clock, sim_inputs)
    calib = build_identity_calibration(config.sensor.height_px, config.sensor.width_px)
    apps = build_apps(sil_config, bus, clock, drivers, MONITORED_SUBSYSTEMS, calib, uplink_key)
    return SilSystem(
        apps=apps,
        bus=bus,
        clock=clock,
        sensor=cast(SimSensor, drivers.sensor),
        gimbal=cast(SimGimbal, drivers.gimbal),
        station=cast(SimStationLink, drivers.station),
        thermal_sensor=cast(SimScalarSensor, drivers.thermal_sensor),
        power_sensor=cast(SimScalarSensor, drivers.power_sensor),
    )


class SilHarness:
    """Deterministic single-threaded driver for a SilSystem (no scheduler threads)."""

    def __init__(self, system: SilSystem) -> None:
        """Seed the payload control state and the FDIR watchdog entries.

        Args:
            system: The wired SilSystem to drive.
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

        Advancing the shared ManualClock each step is what lets the SimGimbal first-order
        dynamics integrate between steps (it integrates lazily on clock-time elapsed), so
        commanded motion actually moves the gimbal across steps.

        Args:
            count: Number of steps to run.
            dt: Seconds to advance `now` per step.
        """
        now = 0.0
        for _ in range(count):
            now += dt
            self._system.clock.advance(dt)
            self.step(now)
