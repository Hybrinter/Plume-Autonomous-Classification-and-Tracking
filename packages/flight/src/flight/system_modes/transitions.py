"""System-mode transition table and decision function (pure).

The authority shell maps each request or ground command to a ModeRequest and calls decide().
decide() reads only its arguments: no clock, no IDs, no bus, no global state.

Rules, in order:
  1. The target must be one of the five system modes (IDLE, STOW, SAFE, INIT, OPERATE).
  2. A request for the current mode is denied (no new activation).
  3. SAFE is accepted from any other state, from any requester.
  4. In SAFE, only a ground EXIT_SAFE to INIT is accepted, and only while no SAFE-triggering
     fault is active. That activation carries recovery_authorized=True.
  5. EXIT_SAFE outside SAFE is denied.
  6. While the fault safety latch is set, every non-SAFE transition is denied.
  7. Any other transition must appear in MODE_EDGES; everything else is denied.

The system boots into SAFE (the authority's first activation; from no mode only SAFE is
accepted). INIT is entered only by an explicit ground command (EXIT_SAFE from SAFE, SET_MODE
from IDLE) and hands off to IDLE when the subsystems report ready. STOW exits only to IDLE or
SAFE.

Contains:
  - RequestKind: where a request came from (a flight subsystem or a ground command).
  - ModeRequest: the requested target mode plus its kind.
  - SafetyEvidence: the latest fault-published safety evidence.
  - Decision: the pure decision result.
  - SYSTEM_MODES / MODE_EDGES: the five modes and the explicit transition table.
  - decide: map (current mode, request, safety evidence) to a Decision.

Satisfies: REQ-OPER-HIGH-002, REQ-SAFE-HIGH-002, REQ-SAFE-EXIT-001.
"""

from __future__ import annotations

# stdlib
import enum
from dataclasses import dataclass

# internal
from flight.libs.types import FaultCode, SystemMode, TransitionDecision


class RequestKind(enum.Enum):
    """Origin class of a mode request. String values mirror member names."""

    SUBSYSTEM = "SUBSYSTEM"  # SystemModeRequestMsg from a flight subsystem
    SET_MODE = "SET_MODE"  # ground SET_MODE command
    EXIT_SAFE = "EXIT_SAFE"  # ground EXIT_SAFE command (always targets INIT)


@dataclass(frozen=True, slots=True)
class ModeRequest:
    """One request presented to decide().

    Attributes:
        target: The requested system mode.
        kind: The origin class of the request.
    """

    target: SystemMode
    kind: RequestKind


@dataclass(frozen=True, slots=True)
class SafetyEvidence:
    """Fault-published safety evidence the authority holds between ticks.

    Attributes:
        safe_latched: True while the fault app holds its SAFE latch.
        active_faults: SAFE-triggering faults the fault app saw in its latest tick.
    """

    safe_latched: bool = False
    active_faults: tuple[FaultCode, ...] = ()


@dataclass(frozen=True, slots=True)
class Decision:
    """Result of one decide() call.

    Attributes:
        decision: ACCEPTED or DENIED.
        resulting_mode: The active mode after the decision (None if no mode is active yet).
        reason: Human-readable reason for the decision.
        recovery_authorized: True only for an accepted EXIT_SAFE.
    """

    decision: TransitionDecision
    resulting_mode: SystemMode | None
    reason: str
    recovery_authorized: bool = False

    @property
    def accepted(self) -> bool:
        """Return True if the request produced a new activation."""
        return self.decision is TransitionDecision.ACCEPTED


SYSTEM_MODES: frozenset[SystemMode] = frozenset(
    {
        SystemMode.IDLE,
        SystemMode.STOW,
        SystemMode.SAFE,
        SystemMode.INIT,
        SystemMode.OPERATE,
    }
)

MODE_EDGES: frozenset[tuple[SystemMode | None, RequestKind, SystemMode]] = frozenset(
    {
        (SystemMode.INIT, RequestKind.SUBSYSTEM, SystemMode.IDLE),
        (SystemMode.IDLE, RequestKind.SET_MODE, SystemMode.INIT),
        (SystemMode.IDLE, RequestKind.SET_MODE, SystemMode.OPERATE),
        (SystemMode.IDLE, RequestKind.SET_MODE, SystemMode.STOW),
        (SystemMode.OPERATE, RequestKind.SET_MODE, SystemMode.IDLE),
        (SystemMode.OPERATE, RequestKind.SET_MODE, SystemMode.STOW),
        (SystemMode.STOW, RequestKind.SET_MODE, SystemMode.IDLE),
    }
)


def _name(mode: SystemMode | None) -> str:
    """Return a log-friendly name for a mode, with "NONE" before the first activation."""
    return "NONE" if mode is None else mode.value


def _accept(target: SystemMode, reason: str, recovery: bool = False) -> Decision:
    """Build an ACCEPTED decision."""
    return Decision(TransitionDecision.ACCEPTED, target, reason, recovery)


def _deny(current: SystemMode | None, reason: str) -> Decision:
    """Build a DENIED decision that keeps the current mode."""
    return Decision(TransitionDecision.DENIED, current, reason)


def decide(current: SystemMode | None, request: ModeRequest, safety: SafetyEvidence) -> Decision:
    """Decide one mode request against the transition table (pure).

    Args:
        current: The active system mode, or None before the first activation.
        request: The requested target and its origin class.
        safety: The latest fault-published safety evidence.

    Returns:
        A Decision. Accepted decisions carry the new mode; denied decisions keep current.
    """
    target = request.target
    if target not in SYSTEM_MODES:
        return _deny(current, f"{target.value} is not a system mode")
    if target is current:
        return _deny(current, f"{target.value} is already active")
    if target is SystemMode.SAFE:
        return _accept(target, f"SAFE requested from {_name(current)}")
    if current is SystemMode.SAFE:
        if request.kind is not RequestKind.EXIT_SAFE or target is not SystemMode.INIT:
            return _deny(current, "only EXIT_SAFE may leave SAFE")
        if safety.active_faults:
            active = ", ".join(code.value for code in safety.active_faults)
            return _deny(current, f"SAFE condition still active: {active}")
        return _accept(target, "EXIT_SAFE authorized", recovery=True)
    if request.kind is RequestKind.EXIT_SAFE:
        return _deny(current, "EXIT_SAFE is only valid in SAFE")
    if safety.safe_latched:
        return _deny(current, "fault safety latch is set")
    if (current, request.kind, target) in MODE_EDGES:
        return _accept(target, f"{_name(current)} -> {target.value}")
    return _deny(
        current,
        f"no transition {_name(current)} -> {target.value} for {request.kind.value}",
    )
