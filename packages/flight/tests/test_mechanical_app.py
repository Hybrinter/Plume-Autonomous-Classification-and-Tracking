"""MechanicalApp tests: lock-state publication + release with the motion interlock."""

from flight.hal.drivers_sim import SimLaunchLock
from flight.libs.bus import MessageBus
from flight.libs.config import PactConfig
from flight.libs.messages import (
    CommandAckMsg,
    GimbalCommandMsg,
    LaunchLockStateMsg,
    RoutedCommandMsg,
)
from flight.libs.time import ManualClock
from flight.libs.types import (
    AckStatus,
    GimbalCommandMode,
    GimbalState,
    LaunchLockState,
    MessageType,
)
from flight.mechanical.app import MechanicalApp


def _app() -> tuple[MechanicalApp, MessageBus]:
    """Build a MechanicalApp over a SimLaunchLock and a fresh bus."""
    bus = MessageBus()
    app = MechanicalApp.from_config(PactConfig(), bus, ManualClock(), SimLaunchLock())
    return app, bus


def _release(seq: int = 1) -> RoutedCommandMsg:
    """Build a routed RELEASE_LAUNCH_LOCK command targeting mechanical."""
    return RoutedCommandMsg(
        msg_type=MessageType.ROUTED_COMMAND,
        timestamp_utc="t",
        target="mechanical",
        command_id="RELEASE_LAUNCH_LOCK",
        params={"phase": "EXECUTE"},
        source="ground",
        seq=seq,
    )


def _motion() -> GimbalCommandMsg:
    """Build a GimbalCommandMsg indicating active RATE motion."""
    return GimbalCommandMsg(
        msg_type=MessageType.GIMBAL_COMMAND,
        timestamp_utc="t",
        frame_id=1,
        mode=GimbalCommandMode.RATE,
        az_value_deg=1.5,
        el_value_deg=0.0,
        state=GimbalState.TRACKING,
        reason="track",
    )


def test_publishes_lock_state_each_tick() -> None:
    """tick() publishes the current LaunchLockStateMsg (ENGAGED at start)."""
    app, bus = _app()
    states = bus.subscribe(LaunchLockStateMsg)
    app.tick()
    assert states.get_nowait().state is LaunchLockState.ENGAGED


def test_release_when_idle_succeeds() -> None:
    """RELEASE_LAUNCH_LOCK with no gimbal motion releases the lock and acks ACCEPTED."""
    app, bus = _app()
    acks = bus.subscribe(CommandAckMsg)
    states = bus.subscribe(LaunchLockStateMsg)
    bus.publish(_release())
    app.tick()
    ack = acks.get_nowait()
    assert ack.status is AckStatus.ACCEPTED
    assert ack.command_id == "RELEASE_LAUNCH_LOCK"
    assert states.get_nowait().state is LaunchLockState.RELEASED


def test_release_inhibited_while_gimbal_moving() -> None:
    """RELEASE_LAUNCH_LOCK is refused (REJECTED) while a gimbal motion command is in flight."""
    app, bus = _app()
    acks = bus.subscribe(CommandAckMsg)
    states = bus.subscribe(LaunchLockStateMsg)
    bus.publish(_motion())  # gimbal commanded to move this cycle
    bus.publish(_release())
    app.tick()
    ack = acks.get_nowait()
    assert ack.status is AckStatus.REJECTED
    assert states.get_nowait().state is LaunchLockState.ENGAGED  # not released
