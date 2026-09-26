"""Driver-agnostic single-step body for the SIL harness and the GSE in-process backend.

step_once reproduces exactly one deterministic SIL cycle: poll mode changes, catch up
the inner and outer loops to ``now``, optionally bind the simulated world at shutter,
acquire and process one payload frame when the duty gate is due (otherwise drain one
unread frame), apply prior-cycle payload pose commands, sample housekeeping, pump the
ISS bridge, publish per-subsystem liveness heartbeats, then run the FDIR tick. It is
Protocol-typed (ImagingSensor /
GimbalActuator / MessageBus) so both SilHarness and the GSE InProcessBackend can
reuse it without depending on concrete drivers. State (payload ControlState + the
FDIR watchdog entries) is threaded in and out, never held in this module.

Contains:
  - SilCycleBind: optional world evaluate + driver feed after catch-up
  - step_once: run one deterministic SIL cycle over the shared bus and return new state.

Satisfies: REQ-SIM-SIL-001.
"""

from __future__ import annotations

# stdlib
from typing import Protocol, runtime_checkable

# internal
from flight.core.composition import MONITORED_SUBSYSTEMS, SystemApps
from flight.fault.watchdog import WatchdogEntry
from flight.hal.interfaces import GimbalActuator, ImagingSensor
from flight.libs.bus import MessageBus
from flight.libs.messages import HeartbeatMsg
from flight.libs.time import ManualClock
from flight.libs.types import MessageType, Ok
from flight.payload.control import ControlState

from sim.environment.records import EnvSample


@runtime_checkable
class SilCycleBind(Protocol):
    """World evaluate and driver feed run after loop catch-up and before acquire."""

    def pre_step(self, now: float) -> EnvSample:
        """Integrate the plant to the clock, evaluate, and push non-None feed slots."""
        ...


def _catch_up_loops(
    apps: SystemApps,
    now: float,
    payload_state: ControlState,
    safe_commanded: bool,
    safe_cleared: bool,
) -> ControlState:
    """Advance inner and outer loops to ``now`` with interleaved outer ticks.

    A first step with ``last_outer_s is None`` stamps loop origins, runs inner
    through ``now`` so encoder samples exist, then outer, then trailing inner.
    Later steps run inner-then-outer for each ``T_out`` slice, then trailing
    inner to ``now``.
    """
    dt_out = apps.payload.controller.cfg.outer.dt_s
    t_out = payload_state.last_outer_s
    if t_out is None:
        payload_state = apps.payload.advance_inner(payload_state, now)
        payload_state, _ = apps.payload.advance_outer(
            payload_state, now, safe_commanded, safe_cleared
        )
        return apps.payload.advance_inner(payload_state, now)
    while t_out + dt_out <= now + 1e-12:
        t_out = t_out + dt_out
        payload_state = apps.payload.advance_inner(payload_state, t_out)
        payload_state, _ = apps.payload.advance_outer(
            payload_state, t_out, safe_commanded, safe_cleared
        )
        safe_commanded = False
        safe_cleared = False
    return apps.payload.advance_inner(payload_state, now)


def step_once(
    apps: SystemApps,
    sensor: ImagingSensor,
    gimbal: GimbalActuator,
    bus: MessageBus,
    clock: ManualClock,
    now: float,
    payload_state: ControlState,
    fault_entries: dict[str, WatchdogEntry],
    bind: SilCycleBind | None = None,
) -> tuple[ControlState, dict[str, WatchdogEntry]]:
    """Advance every subsystem one deterministic cycle over the shared bus.

    Order: poll mode changes -> per-T_out inner-then-outer catch-up to ``now`` ->
    optional bind.pre_step -> acquire and process one payload frame when the imaging
    duty gate is due, otherwise drain one unread frame -> apply payload pose commands
    from the prior cycle -> ISS bridge pump -> command router -> housekeeping
    handle-commands + sample -> storage/downlink ticks -> heartbeats -> FDIR tick.
    Catch-up before acquire leaves encoder samples through shutter time. Ground pose
    commands routed this cycle apply on a later cycle's catch-up.

    Args:
        apps: The wired SystemApps (payload / fault / iss_iface / thermal / electrical).
        sensor: The imaging sensor Protocol the payload acquires or drains this cycle.
        gimbal: The gimbal actuator Protocol whose position feeds the payload controller.
        bus: The shared in-process MessageBus all apps publish/subscribe on.
        clock: The ManualClock supplying wall-clock timestamps for the heartbeats.
        now: Monotonic seconds for the arbiter and watchdog (advanced by the caller).
        payload_state: The payload ControlState threaded in from the previous cycle.
        fault_entries: The FDIR watchdog entries threaded in from the previous cycle.
        bind: Optional world evaluate + driver feed run after catch-up, before acquire.

    Returns:
        A tuple of the new payload ControlState and the new FDIR watchdog entries.

    Notes:
        Driver-agnostic by construction: it imports only HAL Protocols + apps, never a
        concrete driver, so the GSE in-process backend reuses it verbatim. The body is the
        single source of truth for one SIL cycle; SilHarness.step delegates here.
    """
    safe_commanded, safe_cleared = apps.payload.poll_mode_changes()
    payload_state = _catch_up_loops(apps, now, payload_state, safe_commanded, safe_cleared)
    if bind is not None:
        bind.pre_step(now)
    if apps.payload.capture_this_opportunity():
        acquired = sensor.acquire_frame()
        if isinstance(acquired, Ok):
            pos = gimbal.read_position()
            payload_state, _ = apps.payload.process_frame(
                acquired.value,
                payload_state,
                now,
                gimbal_pos=pos.value if isinstance(pos, Ok) else None,
                safe_commanded=safe_commanded,
                safe_cleared=safe_cleared,
            )
    else:
        drained = sensor.drain_frame()
        if isinstance(drained, Ok):
            pos = gimbal.read_position()
            if isinstance(pos, Ok):
                apps.payload.note_gimbal_feedback(pos.value)
    apps.payload.handle_commands()

    apps.iss_iface.tick()
    apps.command_router.tick()

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
