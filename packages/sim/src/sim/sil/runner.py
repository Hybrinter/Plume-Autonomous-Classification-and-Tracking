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
from dataclasses import dataclass

# internal
from flight.core.composition import MONITORED_SUBSYSTEMS, Drivers, SystemApps, build_apps
from flight.fault.watchdog import WatchdogEntry
from flight.hal.drivers_sim import SimGimbal, SimScalarSensor, SimSensor, SimStationLink
from flight.libs.bus import MessageBus
from flight.libs.config import PactConfig
from flight.libs.messages import CommandMsg, HeartbeatMsg
from flight.libs.time import ManualClock
from flight.libs.types import GimbalState, MessageType, MosaicFrame, Ok
from flight.payload.calibration_io import build_identity_calibration
from flight.payload.control import ControlState
from flight.payload.model import ScriptedDetector


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
    inbound_commands: list[CommandMsg],
    thermal_readings: list[float],
    power_readings: list[float],
) -> SilSystem:
    """Construct the sim drivers and wire the flight apps over a fresh bus via build_apps.

    Args:
        config: The PactConfig to wire the apps with.
        clock: The ManualClock shared by all apps (timestamps; the harness advances `now`).
        frames: Raw mosaic frames the SimSensor replays.
        detector: The ScriptedDetector backing the payload.
        inbound_commands: Commands the SimStationLink delivers via the ISS bridge.
        thermal_readings: Temperature readings the thermal sensor replays (Celsius).
        power_readings: Power readings the electrical sensor replays (Watts).

    Returns:
        A SilSystem holding the wired apps, the shared bus/clock, and the sim drivers.
    """
    bus = MessageBus()
    sensor = SimSensor(frames)
    gimbal = SimGimbal(clock=clock, cfg=config.gimbal)
    station = SimStationLink(inbound_commands)
    thermal_sensor = SimScalarSensor(thermal_readings)
    power_sensor = SimScalarSensor(power_readings)
    drivers = Drivers(
        sensor=sensor,
        gimbal=gimbal,
        detector=detector,
        station=station,
        thermal_sensor=thermal_sensor,
        power_sensor=power_sensor,
    )
    calib = build_identity_calibration(config.sensor.height_px, config.sensor.width_px)
    apps = build_apps(config, bus, clock, drivers, MONITORED_SUBSYSTEMS, calib)
    return SilSystem(
        apps=apps,
        bus=bus,
        clock=clock,
        sensor=sensor,
        gimbal=gimbal,
        station=station,
        thermal_sensor=thermal_sensor,
        power_sensor=power_sensor,
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
        """Advance every subsystem one cycle over the shared bus.

        Order: acquire + process one payload frame (if available) -> housekeeping
        handle-commands + sample -> ISS bridge pump -> publish per-subsystem liveness
        heartbeats -> FDIR tick (drains heartbeats + faults, publishes any SAFE).

        Args:
            now: Monotonic seconds for the arbiter and watchdog (advanced by the caller).
        """
        system = self._system
        apps = system.apps

        safe_commanded, safe_cleared = apps.payload.poll_mode_changes()
        acquired = system.sensor.acquire_frame()
        if isinstance(acquired, Ok):
            pos = system.gimbal.read_position()
            self._payload_state, _ = apps.payload.process_frame(
                acquired.value,
                self._payload_state,
                now,
                0.0,
                pos.value if isinstance(pos, Ok) else None,
                safe_commanded,
                safe_cleared,
            )

        apps.thermal.handle_commands()
        apps.thermal.sample()
        apps.electrical.handle_commands()
        apps.electrical.sample()

        apps.iss_iface.tick()

        for subsystem in MONITORED_SUBSYSTEMS:
            system.bus.publish(
                HeartbeatMsg(
                    msg_type=MessageType.HEARTBEAT,
                    timestamp_utc=system.clock.wall_clock_iso(),
                    subsystem=subsystem,
                    sequence=0,
                )
            )

        self._fault_entries = apps.fault.tick(self._fault_entries, now)

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
