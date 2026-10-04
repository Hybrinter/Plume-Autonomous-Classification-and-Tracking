"""System-mode authority shell tests: requests, commands, activations, sync, and ACKs."""

import math
import threading
import time
from dataclasses import replace

import pytest
from flight.libs.bus import MessageBus, Subscription
from flight.libs.config import PactConfig
from flight.libs.messages import (
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
    ActivationKey,
    FaultCode,
    MessageType,
    ModeTransitionDecision,
    SystemMode,
)
from flight.system_modes.app import SUBSYSTEM, SystemModesApp
from flight.system_modes.transitions import SafetyEvidence

_EPOCH = "test-epoch"


def _app() -> tuple[SystemModesApp, MessageBus]:
    """Build an authority over a fresh bus and ManualClock."""
    bus = MessageBus()
    return SystemModesApp.from_config(PactConfig(), bus, ManualClock(), _EPOCH), bus


def _request(
    mode: SystemMode,
    request_id: str,
    by: str = "test",
    reason: str = "test",
    activation_key: ActivationKey | None = None,
) -> SystemModeRequestMsg:
    """Build a subsystem SystemModeRequestMsg."""
    return SystemModeRequestMsg(
        msg_type=MessageType.SYSTEM_MODE_REQUEST,
        timestamp_utc="t",
        request_id=request_id,
        requested_mode=mode,
        requested_by=by,
        reason=reason,
        activation_key=activation_key,
    )


def _init_complete(request_id: str, sequence: int) -> SystemModeRequestMsg:
    """Build the payload's keyed INIT completion request."""
    return _request(
        SystemMode.IDLE,
        request_id,
        by="payload",
        reason="graph_intent:init_complete",
        activation_key=ActivationKey(_EPOCH, sequence),
    )


def _routed(
    command_id: str, params: dict[str, str | int | float | bool], seq: int
) -> RoutedCommandMsg:
    """Build a RoutedCommandMsg targeting the authority."""
    return RoutedCommandMsg(
        msg_type=MessageType.ROUTED_COMMAND,
        timestamp_utc="t",
        target=SUBSYSTEM,
        command_id=command_id,
        params=params,
        source="ground",
        seq=seq,
    )


def _safety(
    clock: ManualClock,
    sequence: int,
    latched: bool = False,
    faults: tuple[FaultCode, ...] = (),
    *,
    epoch: str = _EPOCH,
    observed_s: float | None = None,
    recovery_request_id: str | None = None,
) -> SafetyStateMsg:
    """Build a fault-published SafetyStateMsg (defaults: current epoch, observed now)."""
    return SafetyStateMsg(
        msg_type=MessageType.SAFETY_STATE,
        timestamp_utc="t",
        active_faults=faults,
        safe_latched=latched,
        safe_reason=faults[0] if faults else FaultCode.NONE,
        evidence_epoch=epoch,
        evidence_sequence=sequence,
        observed_s=clock.monotonic_s() if observed_s is None else observed_s,
        recovery_request_id=recovery_request_id,
    )


def _drain[T](sub: Subscription[T]) -> list[T]:
    """Drain a subscription into a list."""
    out: list[T] = []
    while not sub.empty():
        out.append(sub.get_nowait())
    return out


def _booted() -> tuple[SystemModesApp, MessageBus]:
    """Build an authority and run its boot tick (active mode SAFE, sequence 1)."""
    app, bus = _app()
    app.tick()
    return app, bus


def _fresh_evidence(app: SystemModesApp, bus: MessageBus) -> None:
    """Publish one fresh, unlatched safety record and let the authority accept it."""
    assert isinstance(app.clock, ManualClock)
    bus.publish(_safety(app.clock, app.state.evidence_sequence + 1))


def _init() -> tuple[SystemModesApp, MessageBus]:
    """Boot SAFE, then a fresh-evidence EXIT_SAFE -> INIT (sequence 2)."""
    app, bus = _booted()
    assert isinstance(app.clock, ManualClock)
    _fresh_evidence(app, bus)
    bus.publish(_routed("EXIT_SAFE", {"phase": "EXECUTE"}, seq=100))
    app.tick()
    assert app.state.mode is SystemMode.INIT
    return app, bus


def _idle() -> tuple[SystemModesApp, MessageBus]:
    """Boot SAFE, EXIT_SAFE -> INIT, then a keyed payload completion -> IDLE (sequence 3)."""
    app, bus = _init()
    _fresh_evidence(app, bus)
    bus.publish(_init_complete("ready", sequence=2))
    app.tick()
    assert app.state.mode is SystemMode.IDLE
    return app, bus


def test_first_tick_boots_into_safe() -> None:
    """The first tick publishes one ACCEPTED boot transition and a SAFE activation, sequence 1."""
    app, bus = _app()
    transitions = bus.subscribe(SystemModeTransitionMsg)
    activations = bus.subscribe(SystemModeActivatedMsg)
    app.tick()
    [record] = _drain(transitions)
    [activation] = _drain(activations)
    assert record.decision is ModeTransitionDecision.ACCEPTED
    assert record.previous_mode is None
    assert record.resulting_mode is SystemMode.SAFE
    assert record.activation_sequence == 1
    assert record.request_id == f"{_EPOCH}-boot"
    assert (activation.epoch, activation.sequence) == (_EPOCH, 1)
    assert activation.active_mode is SystemMode.SAFE
    assert not activation.recovery_authorized
    app.tick()
    assert transitions.empty()
    assert activations.empty()


def test_denied_request_publishes_transition_only() -> None:
    """A request with no table edge yields a DENIED record and no activation."""
    app, bus = _booted()
    transitions = bus.subscribe(SystemModeTransitionMsg)
    activations = bus.subscribe(SystemModeActivatedMsg)
    bus.publish(_request(SystemMode.OPERATE, "r1"))
    app.tick()
    [record] = _drain(transitions)
    assert record.decision is ModeTransitionDecision.DENIED
    assert record.activation_sequence is None
    assert activations.empty()
    assert app.state.mode is SystemMode.SAFE


def test_duplicate_requests_are_not_coalesced() -> None:
    """Two SAFE requests in one tick give two records: one ACCEPTED, one DENIED."""
    app, bus = _idle()
    transitions = bus.subscribe(SystemModeTransitionMsg)
    bus.publish(_request(SystemMode.SAFE, "f1", "fault"))
    bus.publish(_request(SystemMode.SAFE, "f2", "fault"))
    app.tick()
    records = _drain(transitions)
    assert [r.decision for r in records] == [
        ModeTransitionDecision.ACCEPTED,
        ModeTransitionDecision.DENIED,
    ]
    assert len({r.transition_id for r in records}) == 2


def test_activation_sequence_is_monotonic() -> None:
    """Each accepted transition takes the next sequence in the epoch."""
    app, bus = _app()
    activations = bus.subscribe(SystemModeActivatedMsg)
    app.tick()
    _fresh_evidence(app, bus)
    bus.publish(_routed("EXIT_SAFE", {"phase": "EXECUTE"}, seq=1))
    app.tick()
    _fresh_evidence(app, bus)
    bus.publish(_init_complete("ready", sequence=2))
    app.tick()
    _fresh_evidence(app, bus)
    bus.publish(_routed("SET_MODE", {"mode": "OPERATE"}, seq=2))
    app.tick()
    activations_seen = _drain(activations)
    assert [(a.epoch, a.sequence) for a in activations_seen] == [(_EPOCH, n) for n in (1, 2, 3, 4)]
    assert [a.active_mode for a in activations_seen] == [
        SystemMode.SAFE,
        SystemMode.INIT,
        SystemMode.IDLE,
        SystemMode.OPERATE,
    ]


def test_sync_replays_snapshot_without_new_sequence() -> None:
    """A sync request under the matching epoch republishes the current activation."""
    app, bus = _booted()
    original = app.state.active
    activations = bus.subscribe(SystemModeActivatedMsg)
    transitions = bus.subscribe(SystemModeTransitionMsg)
    bus.publish(
        SystemModeSyncRequestMsg(
            msg_type=MessageType.SYSTEM_MODE_SYNC_REQUEST,
            timestamp_utc="t",
            request_id="sync-1",
            subscriber="payload",
            expected_epoch=_EPOCH,
            last_sequence=None,
        )
    )
    app.tick()
    [replay] = _drain(activations)
    assert replay == original
    assert transitions.empty()
    assert app.state.sequence == 1


def test_sync_wrong_epoch_is_ignored() -> None:
    """A sync request stamped for another epoch gets no reply."""
    app, bus = _booted()
    activations = bus.subscribe(SystemModeActivatedMsg)
    transitions = bus.subscribe(SystemModeTransitionMsg)
    bus.publish(
        SystemModeSyncRequestMsg(
            msg_type=MessageType.SYSTEM_MODE_SYNC_REQUEST,
            timestamp_utc="t",
            request_id="sync-foreign",
            subscriber="payload",
            expected_epoch="other-epoch",
            last_sequence=None,
        )
    )
    app.tick()
    assert activations.empty()
    assert transitions.empty()


def test_set_mode_command_is_acked_with_correlation() -> None:
    """An accepted SET_MODE is ACKed ACCEPTED with the command's source and seq."""
    app, bus = _idle()
    _fresh_evidence(app, bus)
    acks = bus.subscribe(CommandAckMsg)
    bus.publish(_routed("SET_MODE", {"mode": "OPERATE"}, seq=7))
    app.tick()
    [ack] = _drain(acks)
    assert ack.status is AckStatus.ACCEPTED
    assert (ack.source, ack.seq, ack.command_id) == ("ground", 7, "SET_MODE")
    assert app.state.mode is SystemMode.OPERATE


def test_gimbal_stow_command_is_acked_with_correlation() -> None:
    """GIMBAL_STOW maps to a SET_MODE STOW request and is ACKed on activation."""
    app, bus = _idle()
    _fresh_evidence(app, bus)
    acks = bus.subscribe(CommandAckMsg)
    activations = bus.subscribe(SystemModeActivatedMsg)
    bus.publish(_routed("GIMBAL_STOW", {}, seq=9))
    app.tick()
    [ack] = _drain(acks)
    [activation] = _drain(activations)
    assert ack.status is AckStatus.ACCEPTED
    assert (ack.source, ack.seq, ack.command_id) == ("ground", 9, "GIMBAL_STOW")
    assert activation.active_mode is SystemMode.STOW


def test_gimbal_stow_outside_allowed_edge_is_rejected() -> None:
    """GIMBAL_STOW from SAFE cannot leave SAFE (only EXIT_SAFE may)."""
    app, bus = _booted()
    _fresh_evidence(app, bus)
    acks = bus.subscribe(CommandAckMsg)
    bus.publish(_routed("GIMBAL_STOW", {}, seq=9))
    app.tick()
    [ack] = _drain(acks)
    assert ack.status is AckStatus.REJECTED
    assert app.state.mode is SystemMode.SAFE


def test_invalid_set_mode_is_rejected_without_transition() -> None:
    """An unknown mode name is NACKed and never reaches the transition table."""
    app, bus = _booted()
    acks = bus.subscribe(CommandAckMsg)
    transitions = bus.subscribe(SystemModeTransitionMsg)
    bus.publish(_routed("SET_MODE", {"mode": "WARP"}, seq=1))
    app.tick()
    [ack] = _drain(acks)
    assert ack.status is AckStatus.REJECTED
    assert ack.fault_code is FaultCode.COMMAND_INVALID
    assert transitions.empty()


@pytest.mark.parametrize("legacy", ["ACTIVE", "SCAN", "MODEL_UPLINK", "DATA_DOWNLINK"])
def test_legacy_mode_names_are_never_activated(legacy: str) -> None:
    """Pre-cutover mode names are NACKed without a transition record."""
    app, bus = _idle()
    _fresh_evidence(app, bus)
    acks = bus.subscribe(CommandAckMsg)
    transitions = bus.subscribe(SystemModeTransitionMsg)
    activations = bus.subscribe(SystemModeActivatedMsg)
    bus.publish(_routed("SET_MODE", {"mode": legacy}, seq=1))
    app.tick()
    [ack] = _drain(acks)
    assert ack.status is AckStatus.REJECTED
    assert transitions.empty()
    assert activations.empty()


def test_set_mode_non_string_mode_is_rejected() -> None:
    """A SET_MODE whose mode param is not a string is NACKed (no coercion)."""
    app, bus = _idle()
    _fresh_evidence(app, bus)
    acks = bus.subscribe(CommandAckMsg)
    bus.publish(_routed("SET_MODE", {"mode": 1}, seq=1))
    app.tick()
    [ack] = _drain(acks)
    assert ack.status is AckStatus.REJECTED


def test_commands_for_other_targets_are_ignored() -> None:
    """Routed commands for other subsystems produce no authority output."""
    app, bus = _booted()
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


@pytest.mark.parametrize("phase", ["ARM", "", "RETRY"])
def test_exit_safe_rejects_non_execute_phases(phase: str) -> None:
    """The authority defensively requires the routed EXECUTE phase for EXIT_SAFE."""
    app, bus = _booted()
    _fresh_evidence(app, bus)
    acks = bus.subscribe(CommandAckMsg)
    activations = bus.subscribe(SystemModeActivatedMsg)
    transitions = bus.subscribe(SystemModeTransitionMsg)
    bus.publish(_routed("EXIT_SAFE", {"phase": phase}, seq=1))
    app.tick()
    [ack] = _drain(acks)
    assert ack.status is AckStatus.REJECTED
    assert activations.empty()
    assert transitions.empty()
    assert app.state.mode is SystemMode.SAFE


def test_exit_safe_refused_while_fault_active_then_authorized() -> None:
    """EXIT_SAFE is NACKed while a SAFE fault is active, then authorizes recovery into INIT."""
    app, bus = _booted()
    acks = bus.subscribe(CommandAckMsg)
    activations = bus.subscribe(SystemModeActivatedMsg)
    assert isinstance(app.clock, ManualClock)
    bus.publish(_safety(app.clock, 1, latched=True, faults=(FaultCode.POWER_OVER_LIMIT,)))
    bus.publish(_routed("EXIT_SAFE", {"phase": "EXECUTE"}, seq=2))
    app.tick()
    [nack] = _drain(acks)
    assert nack.status is AckStatus.REJECTED
    assert activations.empty()

    bus.publish(_safety(app.clock, 2, latched=True))
    bus.publish(_routed("EXIT_SAFE", {"phase": "EXECUTE"}, seq=3))
    app.tick()
    [ack] = _drain(acks)
    [activation] = _drain(activations)
    assert ack.status is AckStatus.ACCEPTED
    assert activation.active_mode is SystemMode.INIT
    assert activation.recovery_authorized


def test_non_safe_transition_fails_closed_without_evidence() -> None:
    """With no safety evidence at all, every non-SAFE transition is denied."""
    app, bus = _booted()
    transitions = bus.subscribe(SystemModeTransitionMsg)
    activations = bus.subscribe(SystemModeActivatedMsg)
    bus.publish(_routed("EXIT_SAFE", {"phase": "EXECUTE"}, seq=1))
    app.tick()
    [record] = _drain(transitions)
    assert record.decision is ModeTransitionDecision.DENIED
    assert "safety evidence" in record.reason
    assert activations.empty()
    assert app.state.mode is SystemMode.SAFE


def test_non_safe_transition_fails_closed_on_stale_evidence() -> None:
    """Evidence older than the watchdog interval cannot release a transition."""
    app, bus = _booted()
    assert isinstance(app.clock, ManualClock)
    bus.publish(_safety(app.clock, 1))
    app.tick()
    app.clock.advance(app.cfg.watchdog_interval_s + 1.0)
    transitions = bus.subscribe(SystemModeTransitionMsg)
    bus.publish(_routed("EXIT_SAFE", {"phase": "EXECUTE"}, seq=1))
    app.tick()
    [record] = _drain(transitions)
    assert record.decision is ModeTransitionDecision.DENIED
    assert app.state.mode is SystemMode.SAFE


@pytest.mark.parametrize(
    ("observed_s", "epoch", "sequence"),
    [
        (math.inf, _EPOCH, 2),
        (math.nan, _EPOCH, 2),
        (100.0, _EPOCH, 2),  # future observation
        (None, "other-epoch", 2),
        (None, _EPOCH, -1),
    ],
)
def test_malformed_evidence_never_replaces_accepted(
    observed_s: float | None, epoch: str, sequence: int
) -> None:
    """Wrong-epoch, negative-sequence, nonfinite, and future evidence are dropped."""
    app, bus = _booted()
    assert isinstance(app.clock, ManualClock)
    bus.publish(_safety(app.clock, 1, faults=()))  # accepted baseline
    app.tick()
    baseline = app.state.evidence
    bus.publish(_safety(app.clock, sequence, latched=True, epoch=epoch, observed_s=observed_s))
    bus.publish(_routed("EXIT_SAFE", {"phase": "EXECUTE"}, seq=1))
    app.tick()
    # The malformed record was dropped; the baseline (fresh, unlatched) governs.
    assert app.state.evidence == baseline
    assert app.state.mode is SystemMode.INIT


def test_replayed_and_older_evidence_cannot_clear_newer_fault() -> None:
    """An older clear record published after a newer fault record cannot undo it."""
    app, bus = _booted()
    assert isinstance(app.clock, ManualClock)
    # Newer fault evidence accepted first (sequence 2 latched+fault), then an
    # older clear record (sequence 1) must not replace it.
    bus.publish(_safety(app.clock, 1))
    bus.publish(_safety(app.clock, 2, latched=True, faults=(FaultCode.CAMERA_STALL,)))
    bus.publish(_safety(app.clock, 1))
    app.tick()
    assert app.state.evidence.active_faults == (FaultCode.CAMERA_STALL,)
    assert app.state.evidence.safe_latched
    transitions = bus.subscribe(SystemModeTransitionMsg)
    bus.publish(_routed("EXIT_SAFE", {"phase": "EXECUTE"}, seq=1))
    app.tick()
    [record] = _drain(transitions)
    assert record.decision is ModeTransitionDecision.DENIED
    assert app.state.mode is SystemMode.SAFE


def test_safe_request_decided_before_same_tick_exit_safe() -> None:
    """A SAFE request is decided before -- and also blocks -- a same-tick EXIT_SAFE."""
    app, bus = _init()
    _fresh_evidence(app, bus)
    transitions = bus.subscribe(SystemModeTransitionMsg)
    acks = bus.subscribe(CommandAckMsg)
    bus.publish(_routed("EXIT_SAFE", {"phase": "EXECUTE"}, seq=1))
    bus.publish(_request(SystemMode.SAFE, "f1", "fault"))
    app.tick()
    records = _drain(transitions)
    # The subsystem SAFE request is decided first; the EXIT_SAFE is still
    # arbitrated and denied by same-tick precedence, with a DENIED record.
    assert len(records) == 2
    assert records[0].requested_mode is SystemMode.SAFE
    assert records[0].decision is ModeTransitionDecision.ACCEPTED
    assert records[1].requested_mode is SystemMode.INIT
    assert records[1].decision is ModeTransitionDecision.DENIED
    [ack] = _drain(acks)
    assert ack.status is AckStatus.REJECTED
    assert app.state.mode is SystemMode.SAFE


def test_routed_set_mode_safe_blocks_same_tick_exit_safe() -> None:
    """A routed SET_MODE SAFE observed in a tick denies that tick's EXIT_SAFE."""
    app, bus = _idle()
    _fresh_evidence(app, bus)
    transitions = bus.subscribe(SystemModeTransitionMsg)
    activations = bus.subscribe(SystemModeActivatedMsg)
    acks = bus.subscribe(CommandAckMsg)
    bus.publish(_routed("SET_MODE", {"mode": "SAFE"}, seq=1))
    bus.publish(_routed("EXIT_SAFE", {"phase": "EXECUTE"}, seq=2))
    app.tick()
    records = _drain(transitions)
    assert len(records) == 2
    assert records[0].requested_mode is SystemMode.SAFE
    assert records[0].decision is ModeTransitionDecision.ACCEPTED
    assert records[1].requested_mode is SystemMode.INIT
    assert records[1].decision is ModeTransitionDecision.DENIED
    # The SAFE command wins: the only activation is IDLE -> SAFE; no INIT reopens.
    [activation] = _drain(activations)
    assert activation.active_mode is SystemMode.SAFE
    [exit_ack] = [a for a in _drain(acks) if a.command_id == "EXIT_SAFE"]
    assert exit_ack.status is AckStatus.REJECTED
    assert exit_ack.seq == 2
    assert app.state.mode is SystemMode.SAFE


def test_denied_repeat_safe_command_still_blocks_same_tick_exit_safe() -> None:
    """A repeated SET_MODE SAFE in SAFE is denied but still holds the tick's EXIT_SAFE."""
    app, bus = _booted()
    _fresh_evidence(app, bus)
    transitions = bus.subscribe(SystemModeTransitionMsg)
    activations = bus.subscribe(SystemModeActivatedMsg)
    bus.publish(_routed("SET_MODE", {"mode": "SAFE"}, seq=1))
    bus.publish(_routed("EXIT_SAFE", {"phase": "EXECUTE"}, seq=2))
    app.tick()
    records = _drain(transitions)
    assert len(records) == 2
    assert records[0].requested_mode is SystemMode.SAFE
    assert records[0].decision is ModeTransitionDecision.DENIED
    assert records[1].requested_mode is SystemMode.INIT
    assert records[1].decision is ModeTransitionDecision.DENIED
    assert activations.empty()
    assert app.state.mode is SystemMode.SAFE


def test_earlier_exit_safe_denied_by_later_set_mode_safe() -> None:
    """EXIT_SAFE queued before SET_MODE SAFE in one tick is denied, with no recovery activation."""
    app, bus = _booted()
    _fresh_evidence(app, bus)
    transitions = bus.subscribe(SystemModeTransitionMsg)
    activations = bus.subscribe(SystemModeActivatedMsg)
    acks = bus.subscribe(CommandAckMsg)
    bus.publish(_routed("EXIT_SAFE", {"phase": "EXECUTE"}, seq=1))
    bus.publish(_routed("SET_MODE", {"mode": "SAFE"}, seq=2))
    app.tick()
    records = _drain(transitions)
    assert len(records) == 2
    assert records[0].requested_mode is SystemMode.INIT
    assert records[0].decision is ModeTransitionDecision.DENIED
    assert "SAFE request decided this tick" in records[0].reason
    assert records[1].requested_mode is SystemMode.SAFE
    assert records[1].decision is ModeTransitionDecision.DENIED
    assert "already active" in records[1].reason
    assert activations.empty()
    assert app.state.mode is SystemMode.SAFE
    [exit_ack] = [a for a in _drain(acks) if a.command_id == "EXIT_SAFE"]
    assert exit_ack.status is AckStatus.REJECTED
    assert exit_ack.seq == 1


def test_fault_then_clear_in_one_drain_denies_recovery() -> None:
    """A fault record in a drain blocks EXIT_SAFE even when a newer clear follows."""
    app, bus = _init()
    _fresh_evidence(app, bus)
    bus.publish(_request(SystemMode.SAFE, "f1", "fault"))
    app.tick()
    assert app.state.mode is SystemMode.SAFE
    assert isinstance(app.clock, ManualClock)

    # Same drain: a fault record, then a higher-sequence clear record. The clear
    # becomes the persisted evidence, but the drain's fault still denies recovery.
    transitions = bus.subscribe(SystemModeTransitionMsg)
    seq0 = app.state.evidence_sequence
    bus.publish(_safety(app.clock, seq0 + 1, latched=True, faults=(FaultCode.POWER_OVER_LIMIT,)))
    bus.publish(_safety(app.clock, seq0 + 2))
    bus.publish(_routed("EXIT_SAFE", {"phase": "EXECUTE"}, seq=3))
    app.tick()
    records = _drain(transitions)
    assert len(records) == 1
    assert records[0].decision is ModeTransitionDecision.DENIED
    assert "SAFE condition still active" in records[0].reason
    assert app.state.mode is SystemMode.SAFE
    assert app.state.evidence == SafetyEvidence()

    # Next tick: a fresh clear record alone re-enables recovery.
    bus.publish(_safety(app.clock, seq0 + 3))
    bus.publish(_routed("EXIT_SAFE", {"phase": "EXECUTE"}, seq=4))
    app.tick()
    mode: SystemMode = app.state.mode
    assert mode is SystemMode.INIT


def test_init_completion_requires_current_payload_key() -> None:
    """INIT -> IDLE accepts only the payload's keyed completion request."""
    app, bus = _init()
    _fresh_evidence(app, bus)
    transitions = bus.subscribe(SystemModeTransitionMsg)
    activations = bus.subscribe(SystemModeActivatedMsg)
    bus.publish(_init_complete("stale-key", sequence=1))  # key of an earlier activation
    bus.publish(
        _request(SystemMode.IDLE, "wrong-caller", by="fault", reason="graph_intent:init_complete")
    )
    bus.publish(_request(SystemMode.IDLE, "missing-key", by="payload", reason="ready"))
    app.tick()
    records = _drain(transitions)
    assert all(r.decision is ModeTransitionDecision.DENIED for r in records)
    assert activations.empty()
    assert app.state.mode is SystemMode.INIT

    _fresh_evidence(app, bus)
    bus.publish(_init_complete("ready", sequence=2))
    app.tick()
    [activation] = _drain(activations)
    assert activation.active_mode is SystemMode.IDLE


def test_stale_completion_rejected_after_init_reentry() -> None:
    """A completion keyed to the previous INIT cannot complete the reentered INIT."""
    app, bus = _init()
    _fresh_evidence(app, bus)
    transitions = bus.subscribe(SystemModeTransitionMsg)
    activations = bus.subscribe(SystemModeActivatedMsg)
    # Reenter INIT via SAFE -> EXIT_SAFE: INIT is now sequence 4.
    bus.publish(_request(SystemMode.SAFE, "f1", "fault"))
    app.tick()
    _fresh_evidence(app, bus)
    bus.publish(_routed("EXIT_SAFE", {"phase": "EXECUTE"}, seq=2))
    app.tick()
    assert app.state.mode is SystemMode.INIT
    assert app.state.sequence == 4
    _drain(transitions)
    _drain(activations)
    # The stale completion keyed to the earlier INIT (sequence 2) is denied.
    bus.publish(_init_complete("stale", sequence=2))
    app.tick()
    [record] = _drain(transitions)
    assert record.decision is ModeTransitionDecision.DENIED
    assert activations.empty()
    assert app.state.mode is SystemMode.INIT


def test_safe_request_exempt_from_key_and_evidence_gates() -> None:
    """A SAFE request always reaches the table, even with no evidence and no key."""
    app, bus = _idle()
    transitions = bus.subscribe(SystemModeTransitionMsg)
    activations = bus.subscribe(SystemModeActivatedMsg)
    bus.publish(_request(SystemMode.SAFE, "f1", "fault"))
    app.tick()
    [record] = _drain(transitions)
    [activation] = _drain(activations)
    assert record.decision is ModeTransitionDecision.ACCEPTED
    assert activation.active_mode is SystemMode.SAFE


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
