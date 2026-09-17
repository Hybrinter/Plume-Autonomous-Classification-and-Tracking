"""Fault-to-mode policy and SAFE-mode message construction (pure).

Replaces the legacy per-FaultCode Callable dispatch table (FAULT_HANDLERS) with an
explicit, statically typed policy: a frozenset of SAFE-triggering FaultCodes plus a
pure decide_mode_change() that returns a ModeChangeMsg(SAFE) for those codes and None
for the rest. This removes dynamic dispatch (a function-pointer table) in favor of a
direct membership test while preserving the exact partition of faults the legacy
handlers used: SAFE-triggering = {INFERENCE_NAN, CAMERA_STALL, THERMAL_OVER_LIMIT,
POWER_OVER_LIMIT, GIMBAL_RUNAWAY, GIMBAL_FAULT, WATCHDOG_EXPIRE, MODEL_CORRUPT,
PROCESS_DIED}; log-and-continue = {NONE, INFERENCE_TIMEOUT, STORAGE_FULL, COMM_TIMEOUT,
COMMAND_CRC_FAIL, COMMAND_AUTH_FAIL, COMMAND_SEQ_ERROR, COMMAND_INVALID}.
GIMBAL_FAULT is included because a driver-level hardware fault may render stowing
impossible and requires loud annunciation. Command-ingress faults are log-and-continue
because a bad/spoofed/replayed command must NACK but must never SAFE the vehicle.

Contains:
  - SAFE_TRIGGERING_FAULTS: the FaultCodes that require a transition to SystemMode.SAFE.
  - enter_safe_mode / enter_init_mode: build SAFE-entry / INIT-entry ModeChangeMsg.
  - decide_mode_change: map a FaultEventMsg to a ModeChangeMsg(SAFE) or None.
  - can_enter_init: True when SAFE is latched and no triggering fault fired this tick.

Satisfies: REQ-SAFE-HIGH-002, REQ-GIMB-HIGH-003, REQ-SAFE-EXIT-001.
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


def enter_init_mode(cleared_by: str, now_iso: str) -> ModeChangeMsg:
    """Build a ModeChangeMsg requesting transition from SAFE to INIT.

    ENTER_INIT requires an explicit ground command; this only constructs the message.

    Args:
        cleared_by: Identifier of the operator/command authorising INIT.
        now_iso: Wall-clock ISO timestamp for the message.

    Returns:
        A ModeChangeMsg with new_mode=SystemMode.INIT.
    """
    return ModeChangeMsg(
        msg_type=MessageType.MODE_CHANGE,
        timestamp_utc=now_iso,
        new_mode=SystemMode.INIT,
        requested_by=f"enter_init:{cleared_by}",
    )


def can_enter_init(safe_latched: bool, safe_fault_seen_this_tick: bool) -> bool:
    """Decide whether a ground ENTER_INIT may leave SAFE (pure).

    Args:
        safe_latched: True if SAFE is currently latched (else there is nothing to leave).
        safe_fault_seen_this_tick: True if any SAFE-triggering fault was observed in the tick
            the ENTER_INIT is being evaluated in.

    Returns:
        True iff SAFE is latched AND no SAFE-triggering fault is currently active.
    """
    return safe_latched and not safe_fault_seen_this_tick


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
