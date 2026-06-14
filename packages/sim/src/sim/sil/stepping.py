"""Driver-agnostic single-step body for the SIL harness and the GSE in-process backend.

step_once reproduces exactly one deterministic SIL cycle: poll mode changes, acquire +
process one payload frame (if available), sample housekeeping, pump the ISS bridge, publish
per-subsystem liveness heartbeats, then run the FDIR tick. It is Protocol-typed
(ImagingSensor / GimbalActuator / MessageBus) so both SilHarness and the GSE InProcessBackend
can reuse it without depending on concrete drivers. State (payload ControlState + the FDIR
watchdog entries) is threaded in and out, never held in this module.

Contains:
  - step_once: run one deterministic SIL cycle over the shared bus and return new state.

Satisfies: REQ-SIM-SIL-001.
"""

from __future__ import annotations

# internal
from flight.core.composition import MONITORED_SUBSYSTEMS, SystemApps
from flight.fault.watchdog import WatchdogEntry
from flight.hal.interfaces import GimbalActuator, ImagingSensor
from flight.libs.bus import MessageBus
from flight.libs.messages import HeartbeatMsg
from flight.libs.time import ManualClock
from flight.libs.types import MessageType, Ok
from flight.payload.control import ControlState


def step_once(
    apps: SystemApps,
    sensor: ImagingSensor,
    gimbal: GimbalActuator,
    bus: MessageBus,
    clock: ManualClock,
    now: float,
    payload_state: ControlState,
    fault_entries: dict[str, WatchdogEntry],
) -> tuple[ControlState, dict[str, WatchdogEntry]]:
    """Advance every subsystem one deterministic cycle over the shared bus.

    Order: poll mode changes -> acquire + process one payload frame (if available) -> ISS
    bridge pump (ingress publishes CommandMsg) -> command router (CommandMsg ->
    RoutedCommandMsg + acks) -> housekeeping handle-commands + sample -> publish per-subsystem
    liveness heartbeats -> FDIR tick (drains heartbeats + faults + routed EXIT_SAFE, publishes
    any SAFE + the SafetyStateMsg). Ingress, routing, and target execution all occur in one
    cycle so a routed command is executed and acked the same step it ingests.

    Args:
        apps: The wired SystemApps (payload / fault / iss_iface / thermal / electrical).
        sensor: The imaging sensor Protocol the payload acquires a frame from this cycle.
        gimbal: The gimbal actuator Protocol whose position feeds the payload controller.
        bus: The shared in-process MessageBus all apps publish/subscribe on.
        clock: The ManualClock supplying wall-clock timestamps for the heartbeats.
        now: Monotonic seconds for the arbiter and watchdog (advanced by the caller).
        payload_state: The payload ControlState threaded in from the previous cycle.
        fault_entries: The FDIR watchdog entries threaded in from the previous cycle.

    Returns:
        A tuple of the new payload ControlState and the new FDIR watchdog entries.

    Notes:
        Driver-agnostic by construction: it imports only HAL Protocols + apps, never a
        concrete driver, so the GSE in-process backend reuses it verbatim. The body is the
        single source of truth for one SIL cycle; SilHarness.step delegates here.
    """
    safe_commanded, safe_cleared = apps.payload.poll_mode_changes()
    acquired = sensor.acquire_frame()
    if isinstance(acquired, Ok):
        pos = gimbal.read_position()
        payload_state, _ = apps.payload.process_frame(
            acquired.value,
            payload_state,
            now,
            0.0,
            pos.value if isinstance(pos, Ok) else None,
            safe_commanded,
            safe_cleared,
        )

    apps.iss_iface.tick()
    apps.command_router.tick()

    apps.thermal.handle_commands()
    apps.thermal.sample()
    apps.electrical.handle_commands()
    apps.electrical.sample()

    for subsystem in MONITORED_SUBSYSTEMS:
        bus.publish(
            HeartbeatMsg(
                msg_type=MessageType.HEARTBEAT,
                timestamp_utc=clock.wall_clock_iso(),
                subsystem=subsystem,
                sequence=0,
            )
        )

    fault_entries = apps.fault.tick(fault_entries, now)
    return payload_state, fault_entries
