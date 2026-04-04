"""System mode state machine — valid transitions and transition guard.

Defines the allowed (current_mode → new_mode) pairs and enforces them in
transition_mode().  All mode changes must pass through this function.

Satisfies: REQ-OPER-HIGH-002.
"""

from __future__ import annotations

# stdlib
from typing import Final

# internal
from pact.types.enums import SystemMode
from pact.types.enums import Ok, Err, Result  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Transition table
# ---------------------------------------------------------------------------


VALID_TRANSITIONS: Final[dict[SystemMode, frozenset[SystemMode]]] = {
    SystemMode.IDLE: frozenset({
        SystemMode.ACTIVE,
        SystemMode.SAFE,
        SystemMode.MODEL_UPLINK,
    }),
    SystemMode.ACTIVE: frozenset({
        SystemMode.IDLE,
        SystemMode.SAFE,
        SystemMode.DATA_DOWNLINK,
    }),
    SystemMode.SCAN: frozenset({
        SystemMode.IDLE,
        SystemMode.ACTIVE,
        SystemMode.SAFE,
    }),
    SystemMode.MODEL_UPLINK: frozenset({
        SystemMode.IDLE,
        SystemMode.SAFE,
    }),
    SystemMode.DATA_DOWNLINK: frozenset({
        SystemMode.IDLE,
        SystemMode.SAFE,
    }),
    SystemMode.SAFE: frozenset({
        SystemMode.IDLE,
    }),
}


# ---------------------------------------------------------------------------
# Transition guard
# ---------------------------------------------------------------------------


def transition_mode(
    current: SystemMode,
    requested: SystemMode,
) -> Result[SystemMode, str]:
    """Validate and apply a mode transition.

    Returns Ok(requested) if the transition is allowed by VALID_TRANSITIONS.
    Returns Err(str) with a human-readable message if the transition is invalid.

    The caller (ops/main.py) is responsible for applying the returned mode and
    logging the transition as a TelemetryEventMsg.
    """
    allowed = VALID_TRANSITIONS.get(current, frozenset())
    if requested in allowed:
        return Ok(requested)
    return Err(
        f"invalid mode transition: {current.value} → {requested.value}; "
        f"allowed targets: {[m.value for m in allowed]}"
    )
