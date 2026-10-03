"""Fault-to-mode-request policy and SAFE recovery gating (pure).

Fault policy never selects or executes a system mode itself: SAFE-triggering
faults produce a SystemModeRequestMsg addressed to the external mode authority,
and the fault-owned latch releases only when an authorized activation record
meets the recovery contract. All identities and request IDs arrive as explicit
arguments; nothing here reads clocks or generates randomness.

Contains:
  - SAFE_TRIGGERING_FAULTS: the FaultCodes that require a SAFE request.
  - enter_safe_request: build the SAFE SystemModeRequestMsg.
  - decide_mode_request: map a FaultEventMsg to a request or None.
  - recovery_authorized: gate an activation record for latch release.

Satisfies: REQ-SAFE-HIGH-002, REQ-GIMB-HIGH-003, REQ-SAFE-EXIT-001.
"""

from __future__ import annotations

from flight.libs.messages import (
    FaultEventMsg,
    SystemModeActivatedMsg,
    SystemModeRequestMsg,
)
from flight.libs.types import FaultCode, MessageType, SystemMode

SAFE_TRIGGERING_FAULTS: frozenset[FaultCode] = frozenset(
    {
        FaultCode.INFERENCE_NAN,
        FaultCode.CAMERA_STALL,
        FaultCode.THERMAL_OVER_LIMIT,
        FaultCode.POWER_OVER_LIMIT,
        FaultCode.GIMBAL_RUNAWAY,
        FaultCode.GIMBAL_FAULT,
        FaultCode.GIMBAL_ENCODER_INVALID,
        FaultCode.GIMBAL_CONTROLLER_ERROR,
        FaultCode.GIMBAL_THERMAL,
        FaultCode.GIMBAL_SAFETY_TIMEOUT,
        FaultCode.GIMBAL_CLOSED_LOOP_LOSS,
        FaultCode.GIMBAL_STALE_FEEDBACK,
        FaultCode.GIMBAL_TIME_MAPPING,
        FaultCode.GIMBAL_DUTY_EXHAUSTED,
        FaultCode.GIMBAL_WATCHDOG_UNCONFIRMED,
        FaultCode.WATCHDOG_EXPIRE,
        FaultCode.MODEL_CORRUPT,
        FaultCode.PROCESS_DIED,
    }
)


def enter_safe_request(reason: FaultCode, now_iso: str, request_id: str) -> SystemModeRequestMsg:
    """Build a SAFE SystemModeRequestMsg for the mode authority.

    Args:
        reason: The FaultCode that triggered SAFE; embedded in the request reason.
        now_iso: Wall-clock ISO timestamp for the message.
        request_id: Unique shell-generated request identity.

    Returns:
        A SystemModeRequestMsg with requested_mode=SystemMode.SAFE.
    """
    return SystemModeRequestMsg(
        msg_type=MessageType.SYSTEM_MODE_REQUEST,
        timestamp_utc=now_iso,
        request_id=request_id,
        requested_mode=SystemMode.SAFE,
        requested_by="fault",
        reason=f"safe_mode_entry:{reason.value}",
    )


def decide_mode_request(
    event: FaultEventMsg, now_iso: str, request_id: str
) -> SystemModeRequestMsg | None:
    """Map a fault event to a SAFE request, or None if it is benign.

    Args:
        event: The FaultEventMsg to evaluate.
        now_iso: Wall-clock ISO timestamp for any produced message.
        request_id: Unique shell-generated request identity.

    Returns:
        A SystemModeRequestMsg(SAFE) if event.fault_code is in
        SAFE_TRIGGERING_FAULTS, else None.
    """
    if event.fault_code in SAFE_TRIGGERING_FAULTS:
        return enter_safe_request(event.fault_code, now_iso, request_id)
    return None


def recovery_authorized(
    activation: SystemModeActivatedMsg,
    *,
    expected_epoch: str,
    last_sequence: int | None,
    request_id_consumed: bool,
    safe_fault_seen_this_tick: bool,
) -> bool:
    """Gate whether an activation record authorizes releasing the SAFE latch.

    Args:
        activation: The accepted authority activation record.
        expected_epoch: The composition-root-injected epoch this app runs under.
        last_sequence: The newest authority sequence already observed, or None.
        request_id_consumed: True if this request_id already spent a release.
        safe_fault_seen_this_tick: True if any SAFE-triggering fault fired in the
            tick the release is evaluated in.

    Returns:
        True only for an authority-approved EXIT_SAFE recovery: the record is
        marked recovery_authorized, carries a nonempty unspent request_id, moves
        from SAFE to IDLE under the current epoch, is strictly newer than the
        last observed sequence, and no SAFE-triggering fault fired this tick.
    """
    if not activation.recovery_authorized:
        return False
    if not activation.request_id or request_id_consumed:
        return False
    if activation.previous_mode is not SystemMode.SAFE:
        return False
    if activation.active_mode is not SystemMode.IDLE:
        return False
    if activation.epoch != expected_epoch:
        return False
    if last_sequence is not None and activation.sequence <= last_sequence:
        return False
    return not safe_fault_seen_this_tick
