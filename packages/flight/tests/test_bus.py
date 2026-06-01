"""Tests for the typed pub/sub message bus."""

from queue import Empty

import pytest
from flight.libs.bus import MessageBus, Subscription
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
