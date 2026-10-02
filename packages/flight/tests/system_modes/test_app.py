"""System-mode authority shell tests: requests, commands, activations, sync, and ACKs."""

import threading
import time
from dataclasses import replace

from flight.libs.bus import MessageBus
from flight.libs.config import PactConfig
from flight.libs.messages import (
    ActivationKey,
    CommandAckMsg,
    HeartbeatMsg,
    RoutedCommandMsg,
    SafetyStateMsg,
    SystemModeActivatedMsg,
    SystemModeRequestMsg,
    SystemModeSyncRequestMsg,
    SystemModeTransitionMsg,
)
from flight.libs.time import ManualClock
from flight.libs.types import (
    AckStatus,
    FaultCode,
    MessageType,
    SystemMode,
    TransitionDecision,
)
from flight.system_modes.app import SUBSYSTEM, SystemModesApp

_EPOCH = "test-epoch"


def _app() -> tuple[SystemModesApp, MessageBus]:
    """Build an authority over a fresh bus and ManualClock."""
    bus = MessageBus()
    return SystemModesApp.from_config(PactConfig(), bus, ManualClock(), _EPOCH), bus


def _request(mode: SystemMode, request_id: str, by: str = "test") -> SystemModeRequestMsg:
    """Build a subsystem SystemModeRequestMsg."""
    return SystemModeRequestMsg(
        msg_type=MessageType.SYSTEM_MODE_REQUEST,
        timestamp_utc="t",
        request_id=request_id,
        requested_mode=mode,
        requested_by=by,
        reason="test",
    )


def _routed(command_id: str, params: dict[str, object], seq: int) -> RoutedCommandMsg:
    """Build a RoutedCommandMsg targeting the authority."""
    return RoutedCommandMsg(
        msg_type=MessageType.ROUTED_COMMAND,
        timestamp_utc="t",
        target=SUBSYSTEM,
        command_id=command_id,
        params=params,  # type: ignore[arg-type]
        source="ground",
        seq=seq,
    )


def _safety(latched: bool, faults: tuple[FaultCode, ...] = ()) -> SafetyStateMsg:
    """Build a fault-published SafetyStateMsg."""
    return SafetyStateMsg(
        msg_type=MessageType.SAFETY_STATE,
        timestamp_utc="t",
        mode=SystemMode.SAFE if latched else SystemMode.IDLE,
        active_faults=faults,
        safe_latched=latched,
        safe_reason=faults[0] if faults else FaultCode.NONE,
    )


def _drain(sub: object) -> list:  # type: ignore[type-arg]
    """Drain a subscription into a list."""
    out: list = []  # type: ignore[type-arg]
    while not sub.empty():  # type: ignore[attr-defined]
        out.append(sub.get_nowait())  # type: ignore[attr-defined]
    return out


def test_accepted_request_publishes_transition_then_activation() -> None:
    """Boot INIT is accepted: one ACCEPTED transition and one activation with sequence 1."""
    app, bus = _app()
    transitions = bus.subscribe(SystemModeTransitionMsg)
    activations = bus.subscribe(SystemModeActivatedMsg)
    bus.publish(_request(SystemMode.INIT, "boot-1", "startup_health_gate"))
    app.tick()
    [record] = _drain(transitions)
    [activation] = _drain(activations)
    assert record.decision is TransitionDecision.ACCEPTED
    assert record.previous_mode is None
    assert record.resulting_mode is SystemMode.INIT
    assert record.activation_sequence == 1
    assert record.request_id == "boot-1"
    assert activation.key == ActivationKey(_EPOCH, 1)
    assert activation.active_mode is SystemMode.INIT
    assert activation.request_id == "boot-1"
    assert not activation.recovery_authorized
    assert app.state.mode is SystemMode.INIT


def test_denied_request_publishes_transition_only() -> None:
    """A request with no table edge yields a DENIED record and no activation."""
    app, bus = _app()
    transitions = bus.subscribe(SystemModeTransitionMsg)
    activations = bus.subscribe(SystemModeActivatedMsg)
    bus.publish(_request(SystemMode.OPERATE, "r1"))
    app.tick()
    [record] = _drain(transitions)
    assert record.decision is TransitionDecision.DENIED
    assert record.activation_sequence is None
    assert activations.empty()
    assert app.state.mode is None


def test_duplicate_requests_are_not_coalesced() -> None:
    """Two SAFE requests in one tick give two records: one ACCEPTED, one DENIED."""
    app, bus = _app()
    transitions = bus.subscribe(SystemModeTransitionMsg)
    bus.publish(_request(SystemMode.SAFE, "f1", "fault"))
    bus.publish(_request(SystemMode.SAFE, "f2", "fault"))
    app.tick()
    records = _drain(transitions)
    assert [r.decision for r in records] == [
        TransitionDecision.ACCEPTED,
        TransitionDecision.DENIED,
    ]
    assert len({r.transition_id for r in records}) == 2


def test_activation_sequence_is_monotonic() -> None:
    """Each accepted transition takes the next sequence in the epoch."""
    app, bus = _app()
    activations = bus.subscribe(SystemModeActivatedMsg)
    bus.publish(_request(SystemMode.INIT, "boot"))
    bus.publish(_request(SystemMode.IDLE, "verified", "payload"))
    app.tick()
    bus.publish(_routed("SET_MODE", {"mode": "OPERATE"}, seq=1))
    app.tick()
    keys = [a.key for a in _drain(activations)]
    assert keys == [ActivationKey(_EPOCH, 1), ActivationKey(_EPOCH, 2), ActivationKey(_EPOCH, 3)]


def test_sync_replays_snapshot_without_new_sequence() -> None:
    """A sync request republishes the current activation with the same key."""
    app, bus = _app()
    activations = bus.subscribe(SystemModeActivatedMsg)
    transitions = bus.subscribe(SystemModeTransitionMsg)
    bus.publish(_request(SystemMode.INIT, "boot"))
    app.tick()
    [original] = _drain(activations)
    _drain(transitions)
    bus.publish(
        SystemModeSyncRequestMsg(
            msg_type=MessageType.SYSTEM_MODE_SYNC_REQUEST,
            timestamp_utc="t",
            request_id="sync-1",
            subscriber="payload",
            expected_epoch=None,
            last_sequence=None,
        )
    )
    app.tick()
    [replay] = _drain(activations)
    assert replay == original
    assert transitions.empty()
    assert app.state.sequence == 1


def test_sync_before_first_activation_publishes_nothing() -> None:
    """With no active mode there is no snapshot to replay."""
    app, bus = _app()
    activations = bus.subscribe(SystemModeActivatedMsg)
    bus.publish(
        SystemModeSyncRequestMsg(
            msg_type=MessageType.SYSTEM_MODE_SYNC_REQUEST,
            timestamp_utc="t",
            request_id="sync-1",
            subscriber="payload",
            expected_epoch=None,
            last_sequence=None,
        )
    )
    app.tick()
    assert activations.empty()


def test_set_mode_command_is_acked_with_correlation() -> None:
    """An accepted SET_MODE is ACKed ACCEPTED with the command's source and seq."""
    app, bus = _app()
    acks = bus.subscribe(CommandAckMsg)
    bus.publish(_request(SystemMode.INIT, "boot"))
    bus.publish(_request(SystemMode.IDLE, "verified"))
    bus.publish(_routed("SET_MODE", {"mode": "OPERATE"}, seq=7))
    app.tick()
    [ack] = _drain(acks)
    assert ack.status is AckStatus.ACCEPTED
    assert (ack.source, ack.seq, ack.command_id) == ("ground", 7, "SET_MODE")
    assert app.state.mode is SystemMode.OPERATE


def test_invalid_set_mode_is_rejected_without_transition() -> None:
    """An unknown mode name is NACKed and never reaches the transition table."""
    app, bus = _app()
    acks = bus.subscribe(CommandAckMsg)
    transitions = bus.subscribe(SystemModeTransitionMsg)
    bus.publish(_routed("SET_MODE", {"mode": "WARP"}, seq=1))
    app.tick()
    [ack] = _drain(acks)
    assert ack.status is AckStatus.REJECTED
    assert ack.fault_code is FaultCode.COMMAND_INVALID
    assert transitions.empty()


def test_commands_for_other_targets_are_ignored() -> None:
    """Routed commands for other subsystems produce no authority output."""
    app, bus = _app()
    acks = bus.subscribe(CommandAckMsg)
    bus.publish(
        RoutedCommandMsg(
            msg_type=MessageType.ROUTED_COMMAND,
            timestamp_utc="t",
            target="thermal",
            command_id="SET_THERMAL_LIMIT",
            params={"limit_c": 40.0},
            source="ground",
            seq=1,
        )
    )
    app.tick()
    assert acks.empty()


def test_exit_safe_refused_while_fault_active_then_authorized() -> None:
    """EXIT_SAFE is NACKed while a SAFE fault is active, then authorizes recovery once clear."""
    app, bus = _app()
    acks = bus.subscribe(CommandAckMsg)
    activations = bus.subscribe(SystemModeActivatedMsg)
    bus.publish(_request(SystemMode.SAFE, "f1", "fault"))
    bus.publish(_safety(True, (FaultCode.POWER_OVER_LIMIT,)))
    app.tick()
    _drain(activations)
    bus.publish(_routed("EXIT_SAFE", {"phase": "EXECUTE"}, seq=2))
    app.tick()
    [nack] = _drain(acks)
    assert nack.status is AckStatus.REJECTED
    assert activations.empty()

    bus.publish(_safety(True))
    bus.publish(_routed("EXIT_SAFE", {"phase": "EXECUTE"}, seq=3))
    app.tick()
    [ack] = _drain(acks)
    [activation] = _drain(activations)
    assert ack.status is AckStatus.ACCEPTED
    assert activation.active_mode is SystemMode.IDLE
    assert activation.recovery_authorized


def test_safe_request_decided_before_same_tick_exit_safe() -> None:
    """A SAFE request and an EXIT_SAFE in one tick end in IDLE only via SAFE first."""
    app, bus = _app()
    transitions = bus.subscribe(SystemModeTransitionMsg)
    bus.publish(_routed("EXIT_SAFE", {"phase": "EXECUTE"}, seq=1))
    bus.publish(_request(SystemMode.SAFE, "f1", "fault"))
    app.tick()
    records = _drain(transitions)
    assert records[0].requested_mode is SystemMode.SAFE
    assert records[0].decision is TransitionDecision.ACCEPTED


def test_run_emits_heartbeat_and_stops() -> None:
    """run() publishes a system_modes heartbeat and exits once stopped."""
    bus = MessageBus()
    clock = ManualClock()
    base = PactConfig()
    cfg = replace(base, fault=replace(base.fault, watchdog_interval_s=0.01))
    app = SystemModesApp.from_config(cfg, bus, clock, _EPOCH)
    heartbeats = bus.subscribe(HeartbeatMsg)
    stop = threading.Event()
    thread = threading.Thread(target=app.run, args=(stop,))
    thread.start()
    deadline = time.monotonic() + 5.0
    beats: list[HeartbeatMsg] = []
    while not beats and time.monotonic() < deadline:
        clock.advance(1.0)
        time.sleep(0.02)
        beats = [h for h in _drain(heartbeats) if h.subsystem == SUBSYSTEM]
    stop.set()
    thread.join(timeout=5.0)
    assert not thread.is_alive()
    assert beats and beats[0].sequence == 0
