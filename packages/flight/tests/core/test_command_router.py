"""CommandRouter shell tests: drains CommandMsg, publishes routed/ack/fault over the bus."""

from flight.core.command_router import CommandRouter
from flight.libs.bus import MessageBus
from flight.libs.config import PactConfig
from flight.libs.messages import (
    CommandAckMsg,
    CommandMsg,
    FaultEventMsg,
    RoutedCommandMsg,
    SafetyStateMsg,
)
from flight.libs.time import ManualClock
from flight.libs.types import AckStatus, FaultCode, MessageType, SystemMode


def _router(bus: MessageBus) -> CommandRouter:
    """Build a CommandRouter over a shared bus + manual clock."""
    return CommandRouter.from_config(PactConfig(), bus, ManualClock())


def _command(
    command_id: str, target: str, params: dict[str, object] | None = None, seq: int = 1
) -> CommandMsg:
    """Build a CommandMsg envelope."""
    return CommandMsg(
        msg_type=MessageType.COMMAND,
        timestamp_utc="t",
        target=target,
        command_id=command_id,
        params=params or {},  # type: ignore[arg-type]
        source="ground",
        seq=seq,
    )


def test_routes_nonhazardous_command_to_target() -> None:
    """A SET_THERMAL_LIMIT command is republished as a RoutedCommandMsg to thermal."""
    bus = MessageBus()
    router = _router(bus)
    routed = bus.subscribe(RoutedCommandMsg)
    bus.publish(_command("SET_THERMAL_LIMIT", "thermal", {"limit_c": 70.0}))
    router.tick()
    msg = routed.get_nowait()
    assert msg.target == "thermal"
    assert msg.command_id == "SET_THERMAL_LIMIT"
    assert msg.timestamp_utc != ""  # shell stamped it


def test_core_ping_acked_directly() -> None:
    """A core-targeted PING is executed by the router with an ACCEPTED ack, no routed msg."""
    bus = MessageBus()
    router = _router(bus)
    acks = bus.subscribe(CommandAckMsg)
    routed = bus.subscribe(RoutedCommandMsg)
    bus.publish(_command("PING", "core"))
    router.tick()
    assert acks.get_nowait().status is AckStatus.ACCEPTED
    assert routed.empty()


def test_unroutable_target_nacks_and_faults() -> None:
    """A command to an unknown target yields a REJECTED ack + COMMAND_UNROUTABLE fault."""
    bus = MessageBus()
    router = _router(bus)
    acks = bus.subscribe(CommandAckMsg)
    faults = bus.subscribe(FaultEventMsg)
    bus.publish(_command("PING", "nowhere"))
    router.tick()
    ack = acks.get_nowait()
    assert ack.status is AckStatus.REJECTED
    assert ack.fault_code is FaultCode.COMMAND_UNROUTABLE
    assert faults.get_nowait().fault_code is FaultCode.COMMAND_UNROUTABLE


def test_hazardous_arm_then_execute_routes_across_ticks() -> None:
    """EXIT_SAFE ARM is acked (no routed msg); a later EXECUTE routes to the fault app."""
    bus = MessageBus()
    router = _router(bus)
    routed = bus.subscribe(RoutedCommandMsg)
    acks = bus.subscribe(CommandAckMsg)

    bus.publish(_command("EXIT_SAFE", "fault", {"phase": "ARM"}, seq=1))
    router.tick()
    assert routed.empty()
    assert acks.get_nowait().status is AckStatus.ACCEPTED

    bus.publish(_command("EXIT_SAFE", "fault", {"phase": "EXECUTE"}, seq=2))
    router.tick()
    msg = routed.get_nowait()
    assert msg.command_id == "EXIT_SAFE"
    assert msg.target == "fault"


def test_safety_state_drained_updates_inhibit_view() -> None:
    """The router tracks the latest SafetyStateMsg.safe_latched flag."""
    bus = MessageBus()
    router = _router(bus)
    bus.publish(
        SafetyStateMsg(
            msg_type=MessageType.SAFETY_STATE,
            timestamp_utc="t",
            mode=SystemMode.SAFE,
            active_faults=(FaultCode.THERMAL_OVER_LIMIT,),
            safe_latched=True,
            safe_reason=FaultCode.THERMAL_OVER_LIMIT,
        )
    )
    router.tick()
    assert router.state.safe_latched is True
