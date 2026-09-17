"""Unit tests for flight.fault.mode -- SystemMode legal edges."""

from flight.fault.mode import (
    ModeEvent,
    ModeEventKind,
    SystemModeState,
    begin_tick,
    initial_mode_state,
    step,
)
from flight.libs.messages import ModeChangeMsg
from flight.libs.types import FaultCode, SystemMode


def _apply(
    kind: ModeEventKind,
    state: SystemModeState | None = None,
    fault: FaultCode = FaultCode.NONE,
) -> tuple[SystemModeState, ModeChangeMsg | None]:
    """Step from boot or a given state."""
    snap = initial_mode_state() if state is None else state
    return step(snap, ModeEvent(kind, "test", fault), "t")


def test_boot_is_latched_safe() -> None:
    """Software starts latched SAFE."""
    state = initial_mode_state()
    assert state.mode is SystemMode.SAFE
    assert state.safe_latched is True


def test_enter_init_from_safe() -> None:
    """ENTER_INIT leaves SAFE for INIT when no fault is live."""
    state, msg = _apply(ModeEventKind.ENTER_INIT)
    assert state.mode is SystemMode.INIT
    assert state.safe_latched is False
    assert msg is not None and msg.new_mode is SystemMode.INIT


def test_enter_init_refused_when_fault_this_tick() -> None:
    """A SAFE-triggering fault this tick blocks ENTER_INIT."""
    state, _ = _apply(ModeEventKind.FAULT, fault=FaultCode.POWER_OVER_LIMIT)
    state = begin_tick(state)
    state, _ = _apply(ModeEventKind.FAULT, state, FaultCode.POWER_OVER_LIMIT)
    held, msg = _apply(ModeEventKind.ENTER_INIT, state)
    assert held.mode is SystemMode.SAFE
    assert msg is None


def test_homing_complete_to_idle_then_operate() -> None:
    """INIT -> IDLE -> OPERATE on homing complete then ENTER_OPERATE."""
    state, _ = _apply(ModeEventKind.ENTER_INIT)
    state, msg = _apply(ModeEventKind.HOMING_COMPLETE, state)
    assert state.mode is SystemMode.IDLE
    assert msg is not None and msg.new_mode is SystemMode.IDLE
    state, msg = _apply(ModeEventKind.ENTER_OPERATE, state)
    assert state.mode is SystemMode.OPERATE
    assert msg is not None and msg.new_mode is SystemMode.OPERATE


def test_model_suspend_resumes_only_if_suspended() -> None:
    """OPERATE preempts to IDLE; commanded IDLE does not auto-resume."""
    state, _ = _apply(ModeEventKind.ENTER_INIT)
    state, _ = _apply(ModeEventKind.HOMING_COMPLETE, state)
    state, _ = _apply(ModeEventKind.ENTER_OPERATE, state)
    state, msg = _apply(ModeEventKind.MODEL_SUSPEND, state)
    assert state.mode is SystemMode.IDLE
    assert state.operate_suspended is True
    assert msg is not None
    state, msg = _apply(ModeEventKind.MODEL_RESUME, state)
    assert state.mode is SystemMode.OPERATE
    assert msg is not None

    state, _ = _apply(ModeEventKind.ENTER_IDLE, state)
    assert state.operate_suspended is False
    held, msg = _apply(ModeEventKind.MODEL_RESUME, state)
    assert held.mode is SystemMode.IDLE
    assert msg is None


def test_stow_completes_to_safe() -> None:
    """IDLE -> STOW -> SAFE with latch set."""
    state, _ = _apply(ModeEventKind.ENTER_INIT)
    state, _ = _apply(ModeEventKind.HOMING_COMPLETE, state)
    state, _ = _apply(ModeEventKind.ENTER_STOW, state)
    assert state.mode is SystemMode.STOW
    state, msg = _apply(ModeEventKind.STOW_COMPLETE, state)
    assert state.mode is SystemMode.SAFE
    assert state.safe_latched is True
    assert msg is not None and msg.new_mode is SystemMode.SAFE


def test_fault_from_operate_latches_safe() -> None:
    """A triggering fault from OPERATE goes to SAFE."""
    state, _ = _apply(ModeEventKind.ENTER_INIT)
    state, _ = _apply(ModeEventKind.HOMING_COMPLETE, state)
    state, _ = _apply(ModeEventKind.ENTER_OPERATE, state)
    state, msg = _apply(ModeEventKind.FAULT, state, FaultCode.WATCHDOG_EXPIRE)
    assert state.mode is SystemMode.SAFE
    assert state.safe_latched is True
    assert msg is not None and msg.new_mode is SystemMode.SAFE
