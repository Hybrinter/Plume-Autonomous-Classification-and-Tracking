"""Driver-agnostic single-step body for the SIL harness and the GSE in-process backend.

step_once reproduces exactly one deterministic SIL cycle: drain activations and
safety evidence, catch up the inner and outer control loops to ``now``, optionally
bind the simulated world at shutter, run one conservative capture cycle (acquire +
process when the policy/duty gate is due, otherwise drain one unread frame), apply
routed payload commands, pump the ISS bridge, run the command router, sample
housekeeping, tick storage/downlink, publish per-subsystem liveness heartbeats,
then run the FDIR tick. It is Protocol-typed (ImagingSensor / GimbalActuator /
MessageBus) so both SilHarness and the GSE InProcessBackend can reuse it without
depending on concrete drivers. State (payload PayloadState + the FDIR watchdog
entries) is threaded in and out, never held in this module.

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
from flight.libs.types import MessageType
from flight.payload.state import PayloadState

from sim.environment.records import EnvSample


@runtime_checkable
class SilCycleBind(Protocol):
    """World evaluate and driver feed run after loop catch-up and before acquire."""

    def pre_step(self, now: float) -> EnvSample:
        """Integrate the plant to the clock, evaluate, and push non-None feed slots."""
        ...


def _catch_up_loops(
    apps: SystemApps,
    clock: ManualClock,
    now: float,
    payload_state: PayloadState,
) -> PayloadState:
    """Advance the shared clock to ``now`` in inner-period control ticks.

    The control loops are seeded once at the current clock, then the clock
    advances to ``now`` in increments of the payload inner period
    (``min(current + dt, now)``, terminating within 1e-12) with the same
    ``control_tick`` seam invoked at each deadline. Physical device time
    therefore progresses even while the actuator is inhibited -- no torque
    writes or forged timestamps are needed to advance simulated plants.
    """
    dt = apps.payload.servo.cfg.inner.dt_s
    state = apps.payload.control_tick(payload_state, clock.monotonic_s())
    t = clock.monotonic_s()
    while t < now:
        t = min(t + dt, now)
        clock.advance(t - clock.monotonic_s())
        state = apps.payload.control_tick(state, clock.monotonic_s())
    return state


def step_once(
    apps: SystemApps,
    sensor: ImagingSensor,
    gimbal: GimbalActuator,
    bus: MessageBus,
    clock: ManualClock,
    now: float,
    payload_state: PayloadState,
    fault_entries: dict[str, WatchdogEntry],
    bind: SilCycleBind | None = None,
) -> tuple[PayloadState, dict[str, WatchdogEntry]]:
    """Advance every subsystem one deterministic cycle over the shared bus.

    Order: system-mode authority tick (boot SAFE plus any queued request is
    decided before the payload's activation drain) -> drain payload
    activations/safety evidence at the current clock
    -> inner-period control ticks that advance the shared clock to ``now``
    (routed payload commands commit inside the outer tick) -> optional
    bind.pre_step (one control-owned feedback sample first, so a non-grid
    shutter has actual encoder evidence) -> one conservative capture cycle
    -> ISS bridge pump -> command router -> authority tick (routed
    SET_MODE/EXIT_SAFE/GIMBAL_STOW decided the same cycle) -> housekeeping
    handle-commands + sample -> storage/downlink ticks -> heartbeats -> FDIR tick
    -> authority tick (fault SAFE requests arbitrated the same cycle).
    This function owns forward advancement of the shared clock to ``now`` and
    never rewinds it; callers must not advance the clock separately. Activations
    are published only by the real system-mode authority (or an explicit
    test-fixture injection via publish_activation); nothing here fabricates
    authority.

    Args:
        apps: The wired SystemApps (payload / fault / iss_iface / thermal / electrical).
        sensor: The imaging sensor Protocol the payload acquires or drains this cycle.
        gimbal: The gimbal actuator Protocol whose position feeds the payload servo.
        bus: The shared in-process MessageBus all apps publish/subscribe on.
        clock: The ManualClock supplying wall-clock timestamps for the heartbeats.
        now: Target monotonic seconds for the graphs and watchdog; step_once
            advances the shared clock forward to it (never backward).
        payload_state: The payload PayloadState threaded in from the previous cycle.
        fault_entries: The FDIR watchdog entries threaded in from the previous cycle.
        bind: Optional world evaluate + driver feed run after catch-up, before acquire.

    Returns:
        A tuple of the new payload PayloadState and the new FDIR watchdog entries.

    Notes:
        Driver-agnostic by construction: it imports only HAL Protocols + apps, never a
        concrete driver, so the GSE in-process backend reuses it verbatim. The body is the
        single source of truth for one SIL cycle; SilHarness.step delegates here.
    """
    apps.system_modes.tick()
    payload_state = apps.payload.poll_activations(payload_state, clock.monotonic_s())
    payload_state = _catch_up_loops(apps, clock, now, payload_state)
    apps.payload.sample_feedback()
    if bind is not None:
        bind.pre_step(now)
    payload_state, _capture = apps.payload.capture_once(payload_state, now)

    apps.iss_iface.tick()
    apps.command_router.tick()
    apps.system_modes.tick()

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
    apps.system_modes.tick()
    return payload_state, fault_entries
