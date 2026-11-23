"""Tests for the pure fault-to-request policy and SAFE recovery gating."""

from flight.fault.policy import (
    SAFE_TRIGGERING_FAULTS,
    decide_mode_request,
    enter_safe_request,
    recovery_authorized,
)
from flight.libs.messages import FaultEventMsg, SystemModeActivatedMsg
from flight.libs.types import FaultCode, MessageType, SystemMode

_EPOCH = "epoch-test"


def _fault(code: FaultCode) -> FaultEventMsg:
    """Build a FaultEventMsg carrying the given fault code."""
    return FaultEventMsg(
        msg_type=MessageType.FAULT_EVENT,
        timestamp_utc="t",
        fault_code=code,
        subsystem="payload",
        detail="",
    )


def _activation(
    *,
    sequence: int = 1,
    previous_mode: SystemMode | None = SystemMode.SAFE,
    active_mode: SystemMode = SystemMode.INIT,
    epoch: str = _EPOCH,
    request_id: str | None = "req-1",
    recovery_authorized: bool = True,
) -> SystemModeActivatedMsg:
    """Build an authority activation record; defaults satisfy the recovery gate."""
    return SystemModeActivatedMsg(
        msg_type=MessageType.SYSTEM_MODE_ACTIVATED,
        timestamp_utc="t",
        epoch=epoch,
        sequence=sequence,
        previous_mode=previous_mode,
        active_mode=active_mode,
        reason="test",
        request_id=request_id,
        recovery_authorized=recovery_authorized,
    )


def test_safe_triggering_fault_maps_to_safe_request() -> None:
    """A SAFE-triggering fault produces a SystemModeRequestMsg requesting SAFE."""
    request = decide_mode_request(_fault(FaultCode.INFERENCE_NAN), now_iso="t", request_id="r1")
    assert request is not None
    assert request.requested_mode is SystemMode.SAFE
    assert request.requested_by == "fault"
    assert request.request_id == "r1"


def test_non_safe_fault_maps_to_none() -> None:
    """Benign faults produce no system-mode request."""
    assert decide_mode_request(_fault(FaultCode.COMM_TIMEOUT), "t", "r1") is None
    assert decide_mode_request(_fault(FaultCode.STORAGE_FULL), "t", "r1") is None


def test_enter_safe_request_fields() -> None:
    """enter_safe_request requests SAFE tagged with the triggering fault code."""
    enter = enter_safe_request(FaultCode.GIMBAL_RUNAWAY, now_iso="t", request_id="r2")
    assert enter.requested_mode is SystemMode.SAFE
    assert enter.requested_by == "fault"
    assert "GIMBAL_RUNAWAY" in enter.reason


def test_safe_triggering_set_membership() -> None:
    """The SAFE-triggering set matches the handler partition."""
    assert FaultCode.PROCESS_DIED in SAFE_TRIGGERING_FAULTS
    assert FaultCode.WATCHDOG_EXPIRE in SAFE_TRIGGERING_FAULTS
    assert FaultCode.NONE not in SAFE_TRIGGERING_FAULTS
    assert FaultCode.INFERENCE_TIMEOUT not in SAFE_TRIGGERING_FAULTS


def test_gimbal_fault_triggers_safe() -> None:
    """A gimbal driver fault routes to SAFE (stow may be impossible; annunciate loudly)."""
    assert FaultCode.GIMBAL_FAULT in SAFE_TRIGGERING_FAULTS


def test_all_xeryon_containment_faults_trigger_safe() -> None:
    """Every Xeryon feedback, controller, duty, and watchdog fault enters SAFE."""
    xeryon_faults = {
        FaultCode.GIMBAL_ENCODER_INVALID,
        FaultCode.GIMBAL_CONTROLLER_ERROR,
        FaultCode.GIMBAL_THERMAL,
        FaultCode.GIMBAL_SAFETY_TIMEOUT,
        FaultCode.GIMBAL_CLOSED_LOOP_LOSS,
        FaultCode.GIMBAL_STALE_FEEDBACK,
        FaultCode.GIMBAL_TIME_MAPPING,
        FaultCode.GIMBAL_DUTY_EXHAUSTED,
        FaultCode.GIMBAL_WATCHDOG_UNCONFIRMED,
    }
    assert xeryon_faults <= SAFE_TRIGGERING_FAULTS


def test_command_ingress_faults_do_not_trigger_safe() -> None:
    """Command CRC/auth/seq/validation faults are annunciated, never SAFE the vehicle."""
    for code in (
        FaultCode.COMMAND_CRC_FAIL,
        FaultCode.COMMAND_AUTH_FAIL,
        FaultCode.COMMAND_SEQ_ERROR,
        FaultCode.COMMAND_INVALID,
    ):
        event = FaultEventMsg(
            msg_type=MessageType.FAULT_EVENT,
            timestamp_utc="2026-01-01T00:00:00.000Z",
            fault_code=code,
            subsystem="iss_iface",
            detail="test",
        )
        assert decide_mode_request(event, "2026-01-01T00:00:00.000Z", "r") is None


def test_recovery_authorized_accepts_clean_record() -> None:
    """A fully valid authorized SAFE->INIT record releases the latch."""
    assert recovery_authorized(
        _activation(),
        expected_epoch=_EPOCH,
        request_id_consumed=False,
        last_sequence=0,
        safe_fault_seen_this_tick=False,
    )


def test_recovery_rejected_without_authorization() -> None:
    """An ordinary IDLE activation never releases the latch."""
    assert not recovery_authorized(
        _activation(recovery_authorized=False),
        expected_epoch=_EPOCH,
        request_id_consumed=False,
        last_sequence=None,
        safe_fault_seen_this_tick=False,
    )


def test_recovery_rejected_on_wrong_request_or_mode() -> None:
    """Missing request_id or wrong previous/active mode is rejected."""
    assert not recovery_authorized(
        _activation(request_id=None),
        expected_epoch=_EPOCH,
        request_id_consumed=False,
        last_sequence=None,
        safe_fault_seen_this_tick=False,
    )
    assert not recovery_authorized(
        _activation(request_id=""),
        expected_epoch=_EPOCH,
        request_id_consumed=False,
        last_sequence=None,
        safe_fault_seen_this_tick=False,
    )
    assert not recovery_authorized(
        _activation(previous_mode=SystemMode.IDLE),
        expected_epoch=_EPOCH,
        request_id_consumed=False,
        last_sequence=None,
        safe_fault_seen_this_tick=False,
    )
    assert not recovery_authorized(
        _activation(active_mode=SystemMode.IDLE),
        expected_epoch=_EPOCH,
        request_id_consumed=False,
        last_sequence=None,
        safe_fault_seen_this_tick=False,
    )
    assert not recovery_authorized(
        _activation(active_mode=SystemMode.OPERATE),
        expected_epoch=_EPOCH,
        request_id_consumed=False,
        last_sequence=None,
        safe_fault_seen_this_tick=False,
    )


def test_recovery_rejected_on_wrong_epoch_or_stale_sequence() -> None:
    """Wrong-epoch and non-increasing sequences are rejected."""
    assert not recovery_authorized(
        _activation(epoch="other"),
        expected_epoch=_EPOCH,
        request_id_consumed=False,
        last_sequence=None,
        safe_fault_seen_this_tick=False,
    )
    assert not recovery_authorized(
        _activation(sequence=3),
        expected_epoch=_EPOCH,
        request_id_consumed=False,
        last_sequence=3,
        safe_fault_seen_this_tick=False,
    )
    assert not recovery_authorized(
        _activation(sequence=2),
        expected_epoch=_EPOCH,
        request_id_consumed=False,
        last_sequence=3,
        safe_fault_seen_this_tick=False,
    )


def test_recovery_rejected_when_safe_fault_fires_this_tick() -> None:
    """A SAFE-triggering fault in the same tick defeats the release."""
    assert not recovery_authorized(
        _activation(),
        expected_epoch=_EPOCH,
        request_id_consumed=False,
        last_sequence=None,
        safe_fault_seen_this_tick=True,
    )
