"""Pure command-routing core: CommandMsg -> dispatch / NACK / ARM-EXECUTE decision.

route_command is a pure function (no bus, no clock, no I/O, no logging): it maps an ingress-
validated CommandMsg plus the routable-target set, the hazardous-command set, the latest
SAFE-latch state, and the armed-state map to a RouteResult describing exactly what the shell
should publish. The shell (flight.core.command_router) owns the bus, the clock, and the mutable
armed-state, and stamps timestamps on the returned messages (built here with timestamp_utc="").

Routing rules (layered authority: iss_iface validates, core routes, actuating apps enforce):
  - target not routable           -> NACK(COMMAND_UNROUTABLE) + unroutable fault event.
  - target == "core"              -> router executes directly (PING/NOOP) -> ACCEPTED ack.
  - non-hazardous routable        -> dispatch RoutedCommandMsg; the target app emits the exec ack.
  - hazardous, phase == "ARM"     -> record armed-state; ACCEPTED("armed") ack, no dispatch.
  - hazardous, phase == "EXECUTE" -> require a prior ARM within arm_window_s AND (for any command
                                     other than EXIT_SAFE) NOT safe_latched; then dispatch and
                                     consume the arm; otherwise NACK.
  - hazardous, any other phase    -> NACK.

Contains:
  - RouteResult: the per-command decision (what to publish + the new armed-state).
  - route_command: run the routing decision for one CommandMsg.

Satisfies: REQ-COMM-CMD-001.
"""

from __future__ import annotations

# stdlib
from dataclasses import dataclass

# internal
from flight.libs.messages import CommandAckMsg, CommandMsg, RoutedCommandMsg
from flight.libs.types import AckStatus, FaultCode, MessageType

_CORE_TARGET = "core"
_EXIT_SAFE = "EXIT_SAFE"


@dataclass(slots=True, frozen=True)
class RouteResult:
    """The outcome of routing one CommandMsg (the shell stamps timestamps + publishes).

    Fields:
        routed_command: RoutedCommandMsg to publish to the target app, or None. Built with
            timestamp_utc="" (the shell stamps it).
        ack: CommandAckMsg to publish (router-level ARM/reject/core-exec ack), or None when the
            target app will emit the execution ack instead. Built with timestamp_utc="".
        unroutable_detail: Detail string for a COMMAND_UNROUTABLE FaultEventMsg the shell must
            emit, or None. Only set on an unroutable target (loud NACK + fault, no silent drop).
        new_armed: The updated armed-state map (keyed by (source, command_id) -> arm time_s).
    """

    routed_command: RoutedCommandMsg | None
    ack: CommandAckMsg | None
    unroutable_detail: str | None
    new_armed: dict[tuple[str, str], float]


def _routed(command: CommandMsg) -> RoutedCommandMsg:
    """Build a RoutedCommandMsg (timestamp stamped by the shell) echoing the command envelope."""
    return RoutedCommandMsg(
        msg_type=MessageType.ROUTED_COMMAND,
        timestamp_utc="",
        target=command.target,
        command_id=command.command_id,
        params=command.params,
        source=command.source,
        seq=command.seq,
    )


def _ack(command: CommandMsg, status: AckStatus, fault: FaultCode, detail: str) -> CommandAckMsg:
    """Build a CommandAckMsg (timestamp stamped by the shell) echoing the command envelope."""
    return CommandAckMsg(
        msg_type=MessageType.COMMAND_ACK,
        timestamp_utc="",
        status=status,
        command_id=command.command_id,
        source=command.source,
        seq=command.seq,
        fault_code=fault,
        detail=detail,
    )


def route_command(
    command: CommandMsg,
    routable_targets: frozenset[str],
    hazardous_ids: frozenset[str],
    safe_latched: bool,
    armed: dict[tuple[str, str], float],
    now: float,
    arm_window_s: float,
) -> RouteResult:
    """Decide how to route one ingress-validated CommandMsg (pure).

    Args:
        command: The validated CommandMsg from ingress (target stamped from the dictionary).
        routable_targets: The set of subsystem targets any command may be routed to.
        hazardous_ids: The opcode strings requiring the ARM/EXECUTE two-step.
        safe_latched: The latest fault-published SAFE-latch state (inhibit pre-check input).
        armed: The current armed-state map ((source, command_id) -> arm monotonic seconds).
        now: Current monotonic seconds (for the ARM expiry check).
        arm_window_s: Seconds an ARM remains valid before EXECUTE must follow.

    Returns:
        A RouteResult describing what the shell should publish and the updated armed-state.
    """
    if command.target not in routable_targets:
        detail = f"unroutable target {command.target!r} for command {command.command_id!r}"
        ack = _ack(command, AckStatus.REJECTED, FaultCode.COMMAND_UNROUTABLE, detail)
        return RouteResult(None, ack, detail, armed)

    if command.target == _CORE_TARGET:
        ack = _ack(command, AckStatus.ACCEPTED, FaultCode.NONE, "executed by core")
        return RouteResult(None, ack, None, armed)

    if command.command_id not in hazardous_ids:
        return RouteResult(_routed(command), None, None, armed)

    key = (command.source, command.command_id)
    phase = str(command.params.get("phase", ""))
    if phase == "ARM":
        new_armed = dict(armed)
        new_armed[key] = now
        ack = _ack(command, AckStatus.ACCEPTED, FaultCode.NONE, "armed")
        return RouteResult(None, ack, None, new_armed)

    if phase == "EXECUTE":
        without_key = {k: v for k, v in armed.items() if k != key}
        armed_at = armed.get(key)
        if armed_at is None or (now - armed_at) > arm_window_s:
            ack = _ack(
                command, AckStatus.REJECTED, FaultCode.COMMAND_INVALID, "execute without valid arm"
            )
            return RouteResult(None, ack, None, without_key)
        if safe_latched and command.command_id != _EXIT_SAFE:
            ack = _ack(
                command, AckStatus.REJECTED, FaultCode.COMMAND_INVALID, "inhibited while SAFE"
            )
            return RouteResult(None, ack, None, without_key)
        return RouteResult(_routed(command), None, None, without_key)

    ack = _ack(
        command,
        AckStatus.REJECTED,
        FaultCode.COMMAND_INVALID,
        "hazardous phase must be ARM/EXECUTE",
    )
    return RouteResult(None, ack, None, armed)
