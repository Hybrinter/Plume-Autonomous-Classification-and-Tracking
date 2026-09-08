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
from flight.libs.messages import HeartbeatMsg, LaunchLockStateMsg
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

    Order: publish current launch-lock snapshot -> poll mode/lock -> apply payload pose
    commands from the prior cycle -> acquire + process one payload frame (if available) ->
    per-T_out inner-then-outer catch-up -> ISS bridge pump -> command router -> mechanical
    tick -> housekeeping handle-commands + sample -> storage/downlink ticks -> heartbeats ->
    FDIR tick. A lock snapshot at the start of the cycle lets fail-closed payload see the
    driver state on step 1. Ground pose commands routed this cycle apply on the next.

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
    lock_read = apps.mechanical.lock.read_state()
    if isinstance(lock_read, Ok):
        bus.publish(
            LaunchLockStateMsg(
                msg_type=MessageType.LAUNCH_LOCK_STATE,
                timestamp_utc=clock.wall_clock_iso(),
                state=lock_read.value,
            )
        )
    safe_commanded, safe_cleared = apps.payload.poll_mode_changes()
    apps.payload.poll_lock_state()
    apps.payload.handle_commands()
    acquired = sensor.acquire_frame()
    if isinstance(acquired, Ok):
        pos = gimbal.read_position()
        slew = 0.0
        payload_state, _ = apps.payload.process_frame(
            acquired.value,
            payload_state,
            now,
            slew,
            pos.value if isinstance(pos, Ok) else None,
            safe_commanded,
            safe_cleared,
        )
    dt_out = apps.payload.controller.cfg.outer.dt_s
    t_out = payload_state.last_outer_s
    if t_out is None:
        payload_state, _ = apps.payload.advance_outer(
            payload_state, now, safe_commanded, safe_cleared
        )
        payload_state = apps.payload.advance_inner(payload_state, now)
    else:
        while t_out + dt_out <= now + 1e-12:
            t_out = t_out + dt_out
            payload_state = apps.payload.advance_inner(payload_state, t_out)
            payload_state, _ = apps.payload.advance_outer(
                payload_state, t_out, safe_commanded, safe_cleared
            )
            safe_commanded = False
            safe_cleared = False
        payload_state = apps.payload.advance_inner(payload_state, now)

    apps.iss_iface.tick()
    apps.command_router.tick()
    apps.mechanical.tick()

    apps.thermal.handle_commands()
    apps.thermal.sample()
    apps.electrical.handle_commands()
    apps.electrical.sample()

    apps.model_deploy.tick()
    apps.storage.tick()
    apps.downlink.tick()

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
