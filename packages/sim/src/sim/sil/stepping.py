"""Driver-agnostic single-step body for the SIL harness and the GSE in-process backend.

step_once reproduces exactly one deterministic SIL cycle: poll mode changes, acquire +
ingest one payload frame (if available), run inner torque ticks and at least one outer
LQG tick over the scenario dt, sample housekeeping, pump the ISS bridge, publish
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
    dt: float = 1.0,
) -> tuple[ControlState, dict[str, WatchdogEntry]]:
    """Advance every subsystem one deterministic cycle over the shared bus.

    Order: poll mode changes -> acquire + ingest one payload frame (if available) ->
    inner/outer control ticks over `dt` (clock advanced per inner chunk) -> ISS
    bridge pump -> command router -> housekeeping -> storage -> downlink -> heartbeats
    -> FDIR tick.

    Args:
        apps: The wired SystemApps (payload / fault / iss_iface / thermal / electrical).
        sensor: The imaging sensor Protocol the payload acquires a frame from this cycle.
        gimbal: The gimbal actuator Protocol whose position feeds the payload controller.
        bus: The shared in-process MessageBus all apps publish/subscribe on.
        clock: The ManualClock supplying wall-clock timestamps for the heartbeats.
        now: Monotonic seconds at the end of this cycle.
        payload_state: The payload ControlState threaded in from the previous cycle.
        fault_entries: The FDIR watchdog entries threaded in from the previous cycle.
        dt: Scenario step duration in seconds. Inner ticks chunk this interval.

    Returns:
        A tuple of the new payload ControlState and the new FDIR watchdog entries.
    """
    del gimbal
    safe_commanded, safe_cleared = apps.payload.poll_mode_changes()
    apps.payload.poll_lock_state()
    acquired = sensor.acquire_frame()
    if isinstance(acquired, Ok):
        payload_state, _ = apps.payload.process_frame(
            acquired.value,
            payload_state,
            now,
            0.0,
            None,
            False,
            False,
        )

    cfg = apps.payload.controller.cfg
    step_dt = dt
    if step_dt <= 0.0:
        n_inner = 0
        chunk = 0.0
        t0 = now
    else:
        n_inner = max(1, int(round(step_dt / cfg.dt_inner_max_s)))
        chunk = step_dt / n_inner
        t0 = now - step_dt
    for i in range(n_inner):
        clock.advance(chunk)
        t_i = t0 + (i + 1) * chunk
        payload_state, _ = apps.payload.apply_control(
            payload_state,
            t_i,
            chunk,
            safe_commanded if i == 0 else False,
            safe_cleared if i == 0 else False,
        )

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
