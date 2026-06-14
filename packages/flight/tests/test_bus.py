"""Tests for the typed pub/sub message bus."""

from queue import Empty

import pytest
from flight.libs.bus import MessageBus, OverflowPolicy, QueuePolicy, Subscription
from flight.libs.messages import HeartbeatMsg, utc_now_iso
from flight.libs.types import MessageType


def _heartbeat(sequence: int) -> HeartbeatMsg:
    """Build a HeartbeatMsg with the given sequence number."""
    return HeartbeatMsg(
        msg_type=MessageType.HEARTBEAT,
        timestamp_utc=utc_now_iso(),
        subsystem="test",
        sequence=sequence,
    )


def test_publish_delivers_to_subscriber() -> None:
    """A subscriber receives a message published for its type."""
    bus = MessageBus()
    sub: Subscription[HeartbeatMsg] = bus.subscribe(HeartbeatMsg)
    bus.publish(_heartbeat(1))
    received = sub.get_nowait()
    assert received.sequence == 1


def test_multiple_subscribers_each_receive() -> None:
    """Every subscriber of a type receives each published message (fan-out)."""
    bus = MessageBus()
    sub_a: Subscription[HeartbeatMsg] = bus.subscribe(HeartbeatMsg)
    sub_b: Subscription[HeartbeatMsg] = bus.subscribe(HeartbeatMsg)
    bus.publish(_heartbeat(7))
    assert sub_a.get_nowait().sequence == 7
    assert sub_b.get_nowait().sequence == 7


def test_no_subscribers_is_noop() -> None:
    """Publishing a type with no subscribers does not raise."""
    bus = MessageBus()
    bus.publish(_heartbeat(1))  # no subscribers; must not raise


def test_subscriber_only_gets_its_type() -> None:
    """A subscription receives only messages of its registered type."""
    bus = MessageBus()
    sub: Subscription[HeartbeatMsg] = bus.subscribe(HeartbeatMsg)
    bus.publish("not a heartbeat")
    assert sub.empty()
    with pytest.raises(Empty):
        sub.get_nowait()


def _drain(sub: Subscription[HeartbeatMsg]) -> list[int]:
    """Drain a heartbeat subscription into its sequence numbers."""
    out: list[int] = []
    while not sub.empty():
        out.append(sub.get_nowait().sequence)
    return out


def test_drop_oldest_discards_oldest_and_counts() -> None:
    """A DROP_OLDEST-bounded queue keeps the newest messages and counts the drops."""
    bus = MessageBus({HeartbeatMsg: QueuePolicy(maxsize=2, overflow=OverflowPolicy.DROP_OLDEST)})
    sub = bus.subscribe(HeartbeatMsg)
    for seq in (1, 2, 3):
        bus.publish(_heartbeat(seq))
    assert _drain(sub) == [2, 3]  # the oldest (1) was dropped
    assert bus.dropped_count(HeartbeatMsg) == 1
    assert bus.total_dropped() == 1


def test_never_drop_keeps_all_and_counts_overflow() -> None:
    """A NEVER_DROP queue never discards a message but counts soft-bound overflow."""
    bus = MessageBus({HeartbeatMsg: QueuePolicy(maxsize=2, overflow=OverflowPolicy.NEVER_DROP)})
    sub = bus.subscribe(HeartbeatMsg)
    for seq in (1, 2, 3):
        bus.publish(_heartbeat(seq))
    assert _drain(sub) == [1, 2, 3]  # nothing dropped
    assert bus.overflow_count(HeartbeatMsg) == 1
    assert bus.dropped_count(HeartbeatMsg) == 0


def test_default_policy_is_unbounded() -> None:
    """With no policy a queue is unbounded (no drops, no overflow counting)."""
    bus = MessageBus()
    sub = bus.subscribe(HeartbeatMsg)
    for seq in range(50):
        bus.publish(_heartbeat(seq))
    assert len(_drain(sub)) == 50
    assert bus.total_dropped() == 0
    assert bus.total_overflow() == 0
