"""CommandRouter shell tests: drains CommandMsg, publishes routed/ack/fault over the bus."""

from flight.core.command_router import CommandRouter
from flight.core.routing import route_command
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
from flight.libs.types import AckStatus, FaultCode, MessageType


def _router(bus: MessageBus) -> CommandRouter:
    """Build a CommandRouter over a shared bus + manual clock."""
    return CommandRouter.from_config(PactConfig(), bus, ManualClock(), "e")


def _command(
    command_id: str,
    target: str,
    params: dict[str, str | int | float | bool] | None = None,
    seq: int = 1,
) -> CommandMsg:
    """Build a CommandMsg envelope."""
    return CommandMsg(
        msg_type=MessageType.COMMAND,
        timestamp_utc="t",
        target=target,
        command_id=command_id,
        params=params or {},
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
    """EXIT_SAFE ARM is acked (no routed msg); a later EXECUTE routes to the mode authority."""
    bus = MessageBus()
    router = _router(bus)
    routed = bus.subscribe(RoutedCommandMsg)
    acks = bus.subscribe(CommandAckMsg)

    bus.publish(_command("EXIT_SAFE", "system_modes", {"phase": "ARM"}, seq=1))
    router.tick()
    assert routed.empty()
    assert acks.get_nowait().status is AckStatus.ACCEPTED

    bus.publish(_command("EXIT_SAFE", "system_modes", {"phase": "EXECUTE"}, seq=2))
    router.tick()
    msg = routed.get_nowait()
    assert msg.command_id == "EXIT_SAFE"
    assert msg.target == "system_modes"


def test_safety_state_drained_updates_inhibit_view() -> None:
    """The router tracks the latest SafetyStateMsg.safe_latched flag."""
    bus = MessageBus()
    router = _router(bus)
    bus.publish(
        SafetyStateMsg(
            msg_type=MessageType.SAFETY_STATE,
            timestamp_utc="t",
            active_faults=(FaultCode.THERMAL_OVER_LIMIT,),
            safe_latched=True,
            safe_reason=FaultCode.THERMAL_OVER_LIMIT,
            evidence_epoch="e",
            evidence_sequence=0,
            observed_s=0.0,
        )
    )
    router.tick()
    assert router.state.safe_latched is True


def _safety(
    *,
    seq: int,
    observed_s: float,
    epoch: str = "e",
    latched: bool = True,
) -> SafetyStateMsg:
    """Build one fault-owned safety evidence record."""
    return SafetyStateMsg(
        msg_type=MessageType.SAFETY_STATE,
        timestamp_utc="t",
        active_faults=(FaultCode.THERMAL_OVER_LIMIT,) if latched else (),
        safe_latched=latched,
        safe_reason=FaultCode.THERMAL_OVER_LIMIT if latched else FaultCode.NONE,
        evidence_epoch=epoch,
        evidence_sequence=seq,
        observed_s=observed_s,
    )


def test_safety_wrong_epoch_cannot_replace_accepted_evidence() -> None:
    """Evidence under a foreign epoch is ignored, keeping the accepted view."""
    bus = MessageBus()
    clock = ManualClock()
    router = CommandRouter.from_config(PactConfig(), bus, clock, "epoch-a")
    bus.publish(_safety(seq=1, observed_s=0.0, epoch="epoch-a"))
    router.tick()
    assert router.state.safe_latched is True
    bus.publish(_safety(seq=2, observed_s=0.0, epoch="epoch-b", latched=False))
    router.tick()
    assert router.state.safe_latched is True
    assert router.state.evidence_sequence == 1


def test_safety_nonincreasing_sequence_cannot_replace() -> None:
    """An older or equal evidence sequence never overwrites the accepted record."""
    bus = MessageBus()
    clock = ManualClock()
    router = CommandRouter.from_config(PactConfig(), bus, clock, "e")
    bus.publish(_safety(seq=5, observed_s=0.0))
    router.tick()
    assert router.state.safe_latched is True
    bus.publish(_safety(seq=4, observed_s=0.0, latched=False))
    bus.publish(_safety(seq=5, observed_s=0.0, latched=False))
    router.tick()
    assert router.state.safe_latched is True
    assert router.state.evidence_sequence == 5


def test_safety_future_and_nonfinite_observation_ignored() -> None:
    """Future or nonfinite observation times cannot become accepted evidence."""
    bus = MessageBus()
    clock = ManualClock()
    router = CommandRouter.from_config(PactConfig(), bus, clock, "e")
    bus.publish(_safety(seq=1, observed_s=100.0))
    bus.publish(_safety(seq=2, observed_s=float("nan")))
    router.tick()
    assert router.state.evidence_sequence == -1
    assert router.state.safe_latched is False


def test_safety_negative_sequence_ignored() -> None:
    """A negative evidence sequence is malformed and ignored."""
    bus = MessageBus()
    clock = ManualClock()
    router = CommandRouter.from_config(PactConfig(), bus, clock, "e")
    bus.publish(_safety(seq=-1, observed_s=0.0))
    router.tick()
    assert router.state.evidence_sequence == -1


def test_stale_safety_blocks_hazardous_execute() -> None:
    """Without fresh safety evidence a hazardous EXECUTE cannot route."""
    routable = frozenset({"payload"})
    hazardous = frozenset({"GIMBAL_GOTO"})
    armed = route_command(
        _command("GIMBAL_GOTO", "payload", {"phase": "ARM"}, seq=1),
        routable,
        hazardous,
        safe_latched=False,
        safety_fresh=True,
        armed={},
        now=0.0,
        arm_window_s=10.0,
    ).new_armed
    result = route_command(
        _command("GIMBAL_GOTO", "payload", {"phase": "EXECUTE"}, seq=2),
        routable,
        hazardous,
        safe_latched=False,
        safety_fresh=False,
        armed=armed,
        now=1.0,
        arm_window_s=10.0,
    )
    assert result.routed_command is None
    assert result.ack is not None
    assert result.ack.status is AckStatus.REJECTED
    rearmed = route_command(
        _command("GIMBAL_GOTO", "payload", {"phase": "ARM"}, seq=3),
        routable,
        hazardous,
        safe_latched=False,
        safety_fresh=True,
        armed={},
        now=0.0,
        arm_window_s=10.0,
    ).new_armed
    latched = route_command(
        _command("GIMBAL_GOTO", "payload", {"phase": "EXECUTE"}, seq=4),
        routable,
        hazardous,
        safe_latched=True,
        safety_fresh=True,
        armed=rearmed,
        now=1.0,
        arm_window_s=10.0,
    )
    assert latched.routed_command is None
    assert latched.ack is not None
    assert latched.ack.status is AckStatus.REJECTED
