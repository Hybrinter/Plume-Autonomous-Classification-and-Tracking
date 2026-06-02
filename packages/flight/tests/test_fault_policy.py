"""Tests for the pure fault-to-mode policy."""

from flight.fault.policy import (
    SAFE_TRIGGERING_FAULTS,
    decide_mode_change,
    enter_safe_mode,
    exit_safe_mode,
)
from flight.libs.messages import FaultEventMsg
from flight.libs.types import FaultCode, MessageType, SystemMode


def _fault(code: FaultCode) -> FaultEventMsg:
    """Build a FaultEventMsg carrying the given fault code."""
    return FaultEventMsg(
        msg_type=MessageType.FAULT_EVENT,
        timestamp_utc="t",
        fault_code=code,
        subsystem="payload",
        detail="",
    )


def test_safe_triggering_fault_maps_to_safe() -> None:
    """A SAFE-triggering fault produces a ModeChangeMsg requesting SAFE."""
    change = decide_mode_change(_fault(FaultCode.INFERENCE_NAN), now_iso="t")
    assert change is not None
    assert change.new_mode is SystemMode.SAFE


def test_non_safe_fault_maps_to_none() -> None:
    """Benign faults produce no mode change."""
    assert decide_mode_change(_fault(FaultCode.COMM_TIMEOUT), now_iso="t") is None
    assert decide_mode_change(_fault(FaultCode.STORAGE_FULL), now_iso="t") is None


def test_enter_and_exit_safe_mode() -> None:
    """enter_safe_mode requests SAFE (tagged with the reason); exit requests IDLE."""
    enter = enter_safe_mode(FaultCode.GIMBAL_RUNAWAY, now_iso="t")
    assert enter.new_mode is SystemMode.SAFE
    assert "GIMBAL_RUNAWAY" in enter.requested_by
    leave = exit_safe_mode("ground_cmd", now_iso="t")
    assert leave.new_mode is SystemMode.IDLE
    assert "ground_cmd" in leave.requested_by


def test_safe_triggering_set_membership() -> None:
    """The SAFE-triggering set matches the legacy handler partition."""
    assert FaultCode.PROCESS_DIED in SAFE_TRIGGERING_FAULTS
    assert FaultCode.WATCHDOG_EXPIRE in SAFE_TRIGGERING_FAULTS
    assert FaultCode.NONE not in SAFE_TRIGGERING_FAULTS
    assert FaultCode.INFERENCE_TIMEOUT not in SAFE_TRIGGERING_FAULTS
