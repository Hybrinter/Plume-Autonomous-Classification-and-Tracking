"""Pure system-mode transition table tests: every (current, kind, target) combination."""

import itertools

import pytest
from flight.libs.types import FaultCode, SystemMode, TransitionDecision
from flight.system_modes.transitions import (
    MODE_EDGES,
    SYSTEM_MODES,
    ModeRequest,
    RequestKind,
    SafetyEvidence,
    decide,
)

_CLEAR = SafetyEvidence()
_LATCHED_CLEAR = SafetyEvidence(safe_latched=True, active_faults=())
_LATCHED_FAULTED = SafetyEvidence(safe_latched=True, active_faults=(FaultCode.POWER_OVER_LIMIT,))

_CURRENTS: tuple[SystemMode | None, ...] = (None, *sorted(SYSTEM_MODES, key=lambda m: m.value))

# The complete set of accepted (current, kind, target) triples with clear, unlatched evidence.
_EXPECTED_ACCEPTED: frozenset[tuple[SystemMode | None, RequestKind, SystemMode]] = frozenset(
    {
        (None, RequestKind.SUBSYSTEM, SystemMode.INIT),
        (SystemMode.INIT, RequestKind.SUBSYSTEM, SystemMode.IDLE),
        (SystemMode.INIT, RequestKind.SET_MODE, SystemMode.STOW),
        (SystemMode.IDLE, RequestKind.SET_MODE, SystemMode.INIT),
        (SystemMode.IDLE, RequestKind.SET_MODE, SystemMode.OPERATE),
        (SystemMode.IDLE, RequestKind.SET_MODE, SystemMode.STOW),
        (SystemMode.OPERATE, RequestKind.SET_MODE, SystemMode.IDLE),
        (SystemMode.OPERATE, RequestKind.SET_MODE, SystemMode.STOW),
        (SystemMode.STOW, RequestKind.SET_MODE, SystemMode.IDLE),
        (SystemMode.STOW, RequestKind.SET_MODE, SystemMode.INIT),
        (SystemMode.SAFE, RequestKind.EXIT_SAFE, SystemMode.IDLE),
    }
    | {
        (current, kind, SystemMode.SAFE)
        for current in _CURRENTS
        if current is not SystemMode.SAFE
        for kind in RequestKind
    }
)


def test_system_modes_are_exactly_the_five() -> None:
    """The authority knows exactly IDLE, STOW, SAFE, INIT, and OPERATE."""
    assert {m.value for m in SYSTEM_MODES} == {"IDLE", "STOW", "SAFE", "INIT", "OPERATE"}


def test_mode_edges_only_use_system_modes() -> None:
    """Every table edge starts (or boots) and ends in a system mode."""
    for current, _kind, target in MODE_EDGES:
        assert current is None or current in SYSTEM_MODES
        assert target in SYSTEM_MODES


@pytest.mark.parametrize(
    ("current", "kind", "target"),
    list(itertools.product(_CURRENTS, RequestKind, SystemMode)),
)
def test_every_combination_matches_table(
    current: SystemMode | None, kind: RequestKind, target: SystemMode
) -> None:
    """With clear evidence, a request is accepted iff it is in the expected table."""
    decision = decide(current, ModeRequest(target, kind), _CLEAR)
    if (current, kind, target) in _EXPECTED_ACCEPTED:
        assert decision.decision is TransitionDecision.ACCEPTED
        assert decision.resulting_mode is target
    else:
        assert decision.decision is TransitionDecision.DENIED
        assert decision.resulting_mode is current
        assert decision.reason
    assert decision.recovery_authorized is (
        (current, kind, target) == (SystemMode.SAFE, RequestKind.EXIT_SAFE, SystemMode.IDLE)
    )


@pytest.mark.parametrize("legacy", ["ACTIVE", "SCAN", "MODEL_UPLINK", "DATA_DOWNLINK"])
def test_legacy_modes_are_never_activated(legacy: str) -> None:
    """The pre-cutover enum members are denied as targets from every state."""
    for current, kind in itertools.product(_CURRENTS, RequestKind):
        decision = decide(current, ModeRequest(SystemMode(legacy), kind), _CLEAR)
        assert decision.decision is TransitionDecision.DENIED
        assert "not a system mode" in decision.reason


@pytest.mark.parametrize("current", _CURRENTS)
def test_safe_wins_even_with_latch_and_active_faults(current: SystemMode | None) -> None:
    """SAFE is accepted from every non-SAFE state regardless of safety evidence."""
    decision = decide(
        current, ModeRequest(SystemMode.SAFE, RequestKind.SUBSYSTEM), _LATCHED_FAULTED
    )
    if current is SystemMode.SAFE:
        assert decision.decision is TransitionDecision.DENIED
    else:
        assert decision.decision is TransitionDecision.ACCEPTED
        assert decision.resulting_mode is SystemMode.SAFE


def test_exit_safe_refused_while_safe_fault_active() -> None:
    """EXIT_SAFE stays in SAFE while a SAFE-triggering fault is still active."""
    decision = decide(
        SystemMode.SAFE, ModeRequest(SystemMode.IDLE, RequestKind.EXIT_SAFE), _LATCHED_FAULTED
    )
    assert decision.decision is TransitionDecision.DENIED
    assert decision.resulting_mode is SystemMode.SAFE
    assert "POWER_OVER_LIMIT" in decision.reason
    assert not decision.recovery_authorized


def test_exit_safe_authorized_with_latch_once_faults_clear() -> None:
    """EXIT_SAFE is accepted into IDLE with recovery_authorized while the latch is still set."""
    decision = decide(
        SystemMode.SAFE, ModeRequest(SystemMode.IDLE, RequestKind.EXIT_SAFE), _LATCHED_CLEAR
    )
    assert decision.decision is TransitionDecision.ACCEPTED
    assert decision.resulting_mode is SystemMode.IDLE
    assert decision.recovery_authorized


def test_set_mode_idle_cannot_leave_safe() -> None:
    """A plain SET_MODE(IDLE) is not a recovery path out of SAFE."""
    decision = decide(SystemMode.SAFE, ModeRequest(SystemMode.IDLE, RequestKind.SET_MODE), _CLEAR)
    assert decision.decision is TransitionDecision.DENIED
    assert "only EXIT_SAFE" in decision.reason


def test_latch_blocks_non_safe_transitions_outside_safe() -> None:
    """While the fault latch is set, the authority refuses every non-SAFE edge."""
    decision = decide(
        SystemMode.IDLE, ModeRequest(SystemMode.OPERATE, RequestKind.SET_MODE), _LATCHED_CLEAR
    )
    assert decision.decision is TransitionDecision.DENIED
    assert "latch" in decision.reason


def test_decide_is_deterministic() -> None:
    """Equal inputs give equal decisions (no hidden state)."""
    request = ModeRequest(SystemMode.OPERATE, RequestKind.SET_MODE)
    assert decide(SystemMode.IDLE, request, _CLEAR) == decide(SystemMode.IDLE, request, _CLEAR)
