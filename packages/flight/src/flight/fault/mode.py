"""Pure SystemMode manager: legal edges, latch, and ModeChangeMsg construction.

The fault app is the only publisher of ModeChangeMsg. This module maps an immutable
SystemModeState plus one event to the next state and an optional mode-change message.

Satisfies: REQ-OPER-HIGH-002, REQ-SAFE-EXIT-001.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from flight.fault.policy import SAFE_TRIGGERING_FAULTS
from flight.libs.messages import ModeChangeMsg
from flight.libs.types import FaultCode, MessageType, SystemMode


class ModeEventKind(Enum):
    """Input discriminant for one mode-manager step."""

    FAULT = "FAULT"
    ENTER_INIT = "ENTER_INIT"
    ENTER_OPERATE = "ENTER_OPERATE"
    ENTER_IDLE = "ENTER_IDLE"
    ENTER_STOW = "ENTER_STOW"
    HOMING_COMPLETE = "HOMING_COMPLETE"
    STOW_COMPLETE = "STOW_COMPLETE"
    HOMING_FAILED = "HOMING_FAILED"
    MODEL_SUSPEND = "MODEL_SUSPEND"
    MODEL_RESUME = "MODEL_RESUME"


@dataclass(frozen=True, slots=True)
class ModeEvent:
    """One input to SystemModeState.step."""

    kind: ModeEventKind
    requested_by: str
    fault_code: FaultCode = FaultCode.NONE


@dataclass(frozen=True, slots=True)
class SystemModeState:
    """Immutable snapshot of the onboard mode machine.

    Fields:
        mode: Current SystemMode.
        safe_latched: True while in SAFE until a successful ENTER_INIT.
        safe_reason: Fault that latched SAFE, or NONE.
        operate_suspended: True when OPERATE was preempted for model activity.
        safe_fault_this_tick: True if a SAFE-triggering fault ran in the current tick.
    """

    mode: SystemMode = SystemMode.SAFE
    safe_latched: bool = True
    safe_reason: FaultCode = FaultCode.NONE
    operate_suspended: bool = False
    safe_fault_this_tick: bool = False


def initial_mode_state() -> SystemModeState:
    """Boot snapshot: latched SAFE with no fault reason."""
    return SystemModeState()


def _change(new_mode: SystemMode, requested_by: str, now_iso: str) -> ModeChangeMsg:
    """Build a ModeChangeMsg for a successful edge."""
    return ModeChangeMsg(
        msg_type=MessageType.MODE_CHANGE,
        timestamp_utc=now_iso,
        new_mode=new_mode,
        requested_by=requested_by,
    )


def _go(
    state: SystemModeState,
    mode: SystemMode,
    requested_by: str,
    now_iso: str,
    *,
    safe_latched: bool | None = None,
    safe_reason: FaultCode | None = None,
    operate_suspended: bool | None = None,
) -> tuple[SystemModeState, ModeChangeMsg]:
    """Apply a transition and emit ModeChangeMsg."""
    latched = state.safe_latched if safe_latched is None else safe_latched
    reason = state.safe_reason if safe_reason is None else safe_reason
    suspended = state.operate_suspended if operate_suspended is None else operate_suspended
    new_state = replace(
        state,
        mode=mode,
        safe_latched=latched,
        safe_reason=reason,
        operate_suspended=suspended,
    )
    return new_state, _change(mode, requested_by, now_iso)


def step(
    state: SystemModeState, event: ModeEvent, now_iso: str
) -> tuple[SystemModeState, ModeChangeMsg | None]:
    """Advance the mode machine by one event.

    Illegal edges return the same state and no message. SAFE-triggering faults win
    from every powered mode.
    """
    if event.kind is ModeEventKind.FAULT:
        if event.fault_code not in SAFE_TRIGGERING_FAULTS:
            return state, None
        marked = replace(state, safe_fault_this_tick=True)
        if marked.mode is SystemMode.SAFE and marked.safe_latched:
            return replace(marked, safe_reason=event.fault_code), None
        return _go(
            marked,
            SystemMode.SAFE,
            event.requested_by,
            now_iso,
            safe_latched=True,
            safe_reason=event.fault_code,
            operate_suspended=False,
        )

    if event.kind is ModeEventKind.HOMING_FAILED:
        if state.mode is SystemMode.SAFE:
            return replace(state, safe_fault_this_tick=True, safe_reason=event.fault_code), None
        return _go(
            replace(state, safe_fault_this_tick=True),
            SystemMode.SAFE,
            event.requested_by,
            now_iso,
            safe_latched=True,
            safe_reason=event.fault_code,
            operate_suspended=False,
        )

    if event.kind is ModeEventKind.ENTER_INIT:
        if state.mode is not SystemMode.SAFE or not state.safe_latched:
            return state, None
        if state.safe_fault_this_tick:
            return state, None
        return _go(
            state,
            SystemMode.INIT,
            event.requested_by,
            now_iso,
            safe_latched=False,
            safe_reason=FaultCode.NONE,
            operate_suspended=False,
        )

    if event.kind is ModeEventKind.HOMING_COMPLETE:
        if state.mode is not SystemMode.INIT:
            return state, None
        return _go(state, SystemMode.IDLE, event.requested_by, now_iso)

    if event.kind is ModeEventKind.ENTER_OPERATE:
        if state.mode is not SystemMode.IDLE:
            return state, None
        return _go(state, SystemMode.OPERATE, event.requested_by, now_iso, operate_suspended=False)

    if event.kind is ModeEventKind.ENTER_IDLE:
        if state.mode is not SystemMode.OPERATE:
            return state, None
        return _go(state, SystemMode.IDLE, event.requested_by, now_iso, operate_suspended=False)

    if event.kind is ModeEventKind.MODEL_SUSPEND:
        if state.mode is SystemMode.OPERATE:
            return _go(state, SystemMode.IDLE, event.requested_by, now_iso, operate_suspended=True)
        return state, None

    if event.kind is ModeEventKind.MODEL_RESUME:
        if state.mode is SystemMode.IDLE and state.operate_suspended:
            return _go(
                state, SystemMode.OPERATE, event.requested_by, now_iso, operate_suspended=False
            )
        return state, None

    if event.kind is ModeEventKind.ENTER_STOW:
        if state.mode is not SystemMode.IDLE:
            return state, None
        return _go(state, SystemMode.STOW, event.requested_by, now_iso, operate_suspended=False)

    if event.kind is ModeEventKind.STOW_COMPLETE:
        if state.mode is not SystemMode.STOW:
            return state, None
        return _go(
            state,
            SystemMode.SAFE,
            event.requested_by,
            now_iso,
            safe_latched=True,
            safe_reason=FaultCode.NONE,
            operate_suspended=False,
        )

    return state, None


def begin_tick(state: SystemModeState) -> SystemModeState:
    """Clear the per-tick SAFE-fault flag at the start of a fault-app tick."""
    return replace(state, safe_fault_this_tick=False)
