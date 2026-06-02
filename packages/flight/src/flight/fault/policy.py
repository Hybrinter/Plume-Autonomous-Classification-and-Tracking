"""Fault-to-mode policy and SAFE-mode message construction (pure).

Replaces the legacy per-FaultCode Callable dispatch table (FAULT_HANDLERS) with an
explicit, statically typed policy: a frozenset of SAFE-triggering FaultCodes plus a
pure decide_mode_change() that returns a ModeChangeMsg(SAFE) for those codes and None
for the rest. This removes dynamic dispatch (a function-pointer table) in favor of a
direct membership test while preserving the exact partition of faults the legacy
handlers used: SAFE-triggering = {INFERENCE_NAN, CAMERA_STALL, THERMAL_OVER_LIMIT,
POWER_OVER_LIMIT, GIMBAL_RUNAWAY, WATCHDOG_EXPIRE, MODEL_CORRUPT, PROCESS_DIED};
log-and-continue = {NONE, INFERENCE_TIMEOUT, STORAGE_FULL, COMM_TIMEOUT}.

Contains:
  - SAFE_TRIGGERING_FAULTS: the FaultCodes that require a transition to SystemMode.SAFE.
  - enter_safe_mode / exit_safe_mode: build SAFE-entry / SAFE-exit ModeChangeMsg.
  - decide_mode_change: map a FaultEventMsg to a ModeChangeMsg(SAFE) or None.

Satisfies: REQ-SAFE-HIGH-002, REQ-GIMB-HIGH-003.
"""

from __future__ import annotations

from flight.libs.messages import FaultEventMsg, ModeChangeMsg
from flight.libs.types import FaultCode, MessageType, SystemMode

SAFE_TRIGGERING_FAULTS: frozenset[FaultCode] = frozenset(
    {
        FaultCode.INFERENCE_NAN,
        FaultCode.CAMERA_STALL,
        FaultCode.THERMAL_OVER_LIMIT,
        FaultCode.POWER_OVER_LIMIT,
        FaultCode.GIMBAL_RUNAWAY,
        FaultCode.WATCHDOG_EXPIRE,
        FaultCode.MODEL_CORRUPT,
        FaultCode.PROCESS_DIED,
    }
)


def enter_safe_mode(reason: FaultCode, now_iso: str) -> ModeChangeMsg:
    """Build a ModeChangeMsg requesting transition to SystemMode.SAFE.

    Args:
        reason: The FaultCode that triggered SAFE entry; embedded in requested_by.
        now_iso: Wall-clock ISO timestamp for the message.

    Returns:
        A ModeChangeMsg with new_mode=SystemMode.SAFE.
    """
    return ModeChangeMsg(
        msg_type=MessageType.MODE_CHANGE,
        timestamp_utc=now_iso,
        new_mode=SystemMode.SAFE,
        requested_by=f"safe_mode_entry:{reason.value}",
    )


def exit_safe_mode(cleared_by: str, now_iso: str) -> ModeChangeMsg:
    """Build a ModeChangeMsg requesting transition out of SAFE to IDLE.

    SAFE exit requires an explicit ground command; this only constructs the message.

    Args:
        cleared_by: Identifier of the operator/command authorising the exit.
        now_iso: Wall-clock ISO timestamp for the message.

    Returns:
        A ModeChangeMsg with new_mode=SystemMode.IDLE.
    """
    return ModeChangeMsg(
        msg_type=MessageType.MODE_CHANGE,
        timestamp_utc=now_iso,
        new_mode=SystemMode.IDLE,
        requested_by=f"safe_mode_exit:{cleared_by}",
    )


def decide_mode_change(event: FaultEventMsg, now_iso: str) -> ModeChangeMsg | None:
    """Map a fault event to a mode-change request, or None if it is benign.

    Args:
        event: The FaultEventMsg to evaluate.
        now_iso: Wall-clock ISO timestamp for any produced message.

    Returns:
        A ModeChangeMsg(SAFE) if event.fault_code is in SAFE_TRIGGERING_FAULTS, else None.
    """
    if event.fault_code in SAFE_TRIGGERING_FAULTS:
        return enter_safe_mode(event.fault_code, now_iso)
    return None
