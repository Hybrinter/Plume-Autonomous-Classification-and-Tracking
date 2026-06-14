"""Pure command-routing core tests: dispatch / reject / ARM-EXECUTE / inhibit."""

from flight.core.routing import route_command
from flight.libs.messages import CommandMsg
from flight.libs.types import AckStatus, FaultCode, MessageType

_ROUTABLE = frozenset({"core", "thermal", "fault", "payload"})
_HAZARDOUS = frozenset({"EXIT_SAFE", "MANUAL_GIMBAL_SLEW"})
_WINDOW = 30.0


def _cmd(
    command_id: str, target: str, params: dict[str, object] | None = None, seq: int = 1
) -> CommandMsg:
    """Build a CommandMsg envelope for routing tests."""
    return CommandMsg(
        msg_type=MessageType.COMMAND,
        timestamp_utc="t",
        target=target,
        command_id=command_id,
        params=params or {},  # type: ignore[arg-type]
        source="ground",
        seq=seq,
    )


def test_unknown_target_rejected_with_fault() -> None:
    """A command whose target is not routable yields a NACK + an unroutable fault."""
    result = route_command(_cmd("PING", "nowhere"), _ROUTABLE, _HAZARDOUS, False, {}, 0.0, _WINDOW)
    assert result.routed_command is None
    assert result.ack is not None
    assert result.ack.status is AckStatus.REJECTED
    assert result.ack.fault_code is FaultCode.COMMAND_UNROUTABLE
    assert result.unroutable_detail is not None


def test_core_target_handled_directly() -> None:
    """A core-targeted command (PING) is executed by the router with an ACCEPTED ack."""
    result = route_command(_cmd("PING", "core"), _ROUTABLE, _HAZARDOUS, False, {}, 0.0, _WINDOW)
    assert result.routed_command is None
    assert result.ack is not None
    assert result.ack.status is AckStatus.ACCEPTED
    assert result.unroutable_detail is None


def test_nonhazardous_routed_without_router_ack() -> None:
    """A non-hazardous routable command is dispatched; the target (not router) acks it."""
    result = route_command(
        _cmd("SET_THERMAL_LIMIT", "thermal", {"limit_c": 70.0}),
        _ROUTABLE,
        _HAZARDOUS,
        False,
        {},
        0.0,
        _WINDOW,
    )
    assert result.routed_command is not None
    assert result.routed_command.target == "thermal"
    assert result.routed_command.command_id == "SET_THERMAL_LIMIT"
    assert result.ack is None


def test_hazardous_arm_then_execute_dispatches() -> None:
    """A hazardous command requires ARM (acked, not dispatched) then EXECUTE (dispatched)."""
    arm = route_command(
        _cmd("EXIT_SAFE", "fault", {"phase": "ARM"}), _ROUTABLE, _HAZARDOUS, False, {}, 0.0, _WINDOW
    )
    assert arm.routed_command is None
    assert arm.ack is not None and arm.ack.status is AckStatus.ACCEPTED
    assert ("ground", "EXIT_SAFE") in arm.new_armed

    execute = route_command(
        _cmd("EXIT_SAFE", "fault", {"phase": "EXECUTE"}),
        _ROUTABLE,
        _HAZARDOUS,
        False,
        arm.new_armed,
        1.0,
        _WINDOW,
    )
    assert execute.routed_command is not None
    assert execute.routed_command.command_id == "EXIT_SAFE"
    assert ("ground", "EXIT_SAFE") not in execute.new_armed


def test_hazardous_execute_without_arm_rejected() -> None:
    """EXECUTE with no prior ARM is rejected (no dispatch)."""
    result = route_command(
        _cmd("EXIT_SAFE", "fault", {"phase": "EXECUTE"}),
        _ROUTABLE,
        _HAZARDOUS,
        False,
        {},
        0.0,
        _WINDOW,
    )
    assert result.routed_command is None
    assert result.ack is not None and result.ack.status is AckStatus.REJECTED


def test_hazardous_execute_after_arm_window_rejected() -> None:
    """An ARM older than arm_window_s no longer authorizes EXECUTE."""
    armed = {("ground", "EXIT_SAFE"): 0.0}
    result = route_command(
        _cmd("EXIT_SAFE", "fault", {"phase": "EXECUTE"}),
        _ROUTABLE,
        _HAZARDOUS,
        False,
        armed,
        _WINDOW + 1.0,
        _WINDOW,
    )
    assert result.routed_command is None
    assert result.ack is not None and result.ack.status is AckStatus.REJECTED


def test_hazardous_execute_inhibited_while_safe() -> None:
    """A non-EXIT_SAFE hazardous EXECUTE is inhibited while SAFE is latched."""
    armed = {("ground", "MANUAL_GIMBAL_SLEW"): 0.0}
    result = route_command(
        _cmd("MANUAL_GIMBAL_SLEW", "payload", {"phase": "EXECUTE"}),
        _ROUTABLE,
        _HAZARDOUS,
        True,  # safe_latched
        armed,
        1.0,
        _WINDOW,
    )
    assert result.routed_command is None
    assert result.ack is not None and result.ack.status is AckStatus.REJECTED
    assert "inhibit" in result.ack.detail.lower()


def test_exit_safe_execute_allowed_while_safe() -> None:
    """EXIT_SAFE is exempt from the SAFE inhibit (it is the recovery command)."""
    armed = {("ground", "EXIT_SAFE"): 0.0}
    result = route_command(
        _cmd("EXIT_SAFE", "fault", {"phase": "EXECUTE"}),
        _ROUTABLE,
        _HAZARDOUS,
        True,  # safe_latched
        armed,
        1.0,
        _WINDOW,
    )
    assert result.routed_command is not None


def test_hazardous_unknown_phase_rejected() -> None:
    """A hazardous command with an unknown phase value is rejected."""
    result = route_command(
        _cmd("EXIT_SAFE", "fault", {"phase": "GO"}), _ROUTABLE, _HAZARDOUS, False, {}, 0.0, _WINDOW
    )
    assert result.routed_command is None
    assert result.ack is not None and result.ack.status is AckStatus.REJECTED
