"""SIL integration: the launch lock inhibits gimbal motion until a hazardous release frees it."""

from flight.libs.commands import build_tc_packet
from flight.libs.config import PactConfig
from flight.libs.messages import CommandAckMsg, LaunchLockStateMsg
from flight.libs.time import ManualClock
from flight.libs.types import AckStatus, LaunchLockState, MessageType, Ok
from sim.scene import build_frames, plume_detector
from sim.sil import SilHarness, build_sil_system

_KEY = b"sil-test-key-0000000000000000000"


def test_launch_lock_inhibits_then_release_frees_the_gimbal() -> None:
    """While ENGAGED the gimbal cannot track the plume; an ARM/EXECUTE release frees it to move."""
    system = build_sil_system(
        PactConfig(),
        ManualClock(),
        build_frames(24),
        plume_detector(),  # a real plume the arbiter wants to track
        inbound_packets=[],
        thermal_readings=[25.0],
        power_readings=[30.0],
        launch_lock_engaged=True,  # start in the launch (locked) configuration
    )
    harness = SilHarness(system)
    acks = system.bus.subscribe(CommandAckMsg)
    lock_states = system.bus.subscribe(LaunchLockStateMsg)

    # Make the payload's first poll see ENGAGED so motion is inhibited from frame one.
    system.bus.publish(
        LaunchLockStateMsg(
            msg_type=MessageType.LAUNCH_LOCK_STATE,
            timestamp_utc="t",
            state=LaunchLockState.ENGAGED,
        )
    )

    now = 0.0

    def advance(steps: int) -> None:
        nonlocal now
        for _ in range(steps):
            now += 1.0
            system.clock.advance(1.0)
            harness.step(now)

    advance(6)  # plume present, but the lock inhibits all gimbal motion
    locked_pos = system.gimbal.read_position()
    assert isinstance(locked_pos, Ok)
    assert abs(locked_pos.value.az_deg) < 0.1  # gimbal held at the origin while ENGAGED
    assert abs(locked_pos.value.el_deg) < 0.1

    # Hazardous release: ARM then EXECUTE over the link.
    system.station.enqueue(
        build_tc_packet("RELEASE_LAUNCH_LOCK", {"phase": "ARM"}, "ground", 1, _KEY, apid=1)
    )
    advance(1)
    system.station.enqueue(
        build_tc_packet("RELEASE_LAUNCH_LOCK", {"phase": "EXECUTE"}, "ground", 2, _KEY, apid=1)
    )
    advance(1)

    release_acks = [
        a
        for a in _drain(acks)
        if a.command_id == "RELEASE_LAUNCH_LOCK" and a.status is AckStatus.ACCEPTED
    ]
    assert release_acks  # the mechanical app accepted the release (gimbal was idle/inhibited)
    latest_lock = [m.state for m in _drain(lock_states)]
    assert latest_lock and latest_lock[-1] is LaunchLockState.RELEASED

    advance(10)  # with the lock released the payload now tracks the plume
    freed_pos = system.gimbal.read_position()
    assert isinstance(freed_pos, Ok)
    assert abs(freed_pos.value.az_deg) > 0.5 or abs(freed_pos.value.el_deg) > 0.5


def _drain(subscription: object) -> list:  # type: ignore[type-arg]
    """Drain all pending messages from a subscription into a list."""
    out: list = []  # type: ignore[type-arg]
    while not subscription.empty():  # type: ignore[attr-defined]
        out.append(subscription.get_nowait())  # type: ignore[attr-defined]
    return out
