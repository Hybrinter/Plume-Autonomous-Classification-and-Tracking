"""Safe mode entry and exit helpers.

Constructs the ModeChangeMsg values that ops/main.py applies when entering or leaving
SAFE mode.  These functions are pure — they produce messages but perform no side effects.
All side effects (stopping processes, disabling actuators) are the responsibility of
ops/main.py when it applies the returned ModeChangeMsg.

Satisfies: REQ-SAFE-HIGH-002.
"""

from __future__ import annotations

# stdlib
from datetime import datetime, timezone

# internal
from pact.types.enums import FaultCode, MessageType, SystemMode
from pact.types.messages import ModeChangeMsg


def enter_safe_mode(reason: FaultCode) -> ModeChangeMsg:
    """Construct a ModeChangeMsg requesting transition to SystemMode.SAFE.

    Args:
        reason: The FaultCode that triggered safe mode entry.  Embedded in
                requested_by so that ops/main.py can log the triggering fault.

    Returns:
        A ModeChangeMsg with new_mode=SystemMode.SAFE.
    """
    return ModeChangeMsg(
        msg_type=MessageType.MODE_CHANGE,
        timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        new_mode=SystemMode.SAFE,
        requested_by=f"safe_mode_entry:{reason.value}",
    )


def exit_safe_mode(cleared_by: str) -> ModeChangeMsg:
    """Construct a ModeChangeMsg requesting transition out of SAFE mode to IDLE.

    Safe mode exit requires an explicit ground command.  This function should only
    be called when the ops process receives a verified ground command to clear the fault.

    Args:
        cleared_by: Identifier of the operator or command that authorised the exit
                    (e.g. "ground_cmd:0x4F2A" or "operator:mission_control").

    Returns:
        A ModeChangeMsg with new_mode=SystemMode.IDLE.

    Note: The VALID_TRANSITIONS table in ops/modes.py allows SAFE → IDLE, so
    ops/main.py's transition_mode() call will succeed.
    """
    return ModeChangeMsg(
        msg_type=MessageType.MODE_CHANGE,
        timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        new_mode=SystemMode.IDLE,
        requested_by=f"safe_mode_exit:{cleared_by}",
    )
