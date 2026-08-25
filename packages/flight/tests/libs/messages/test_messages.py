"""Tests for the migrated message dataclasses."""

import re

import flight.libs.messages as messages
from flight.libs.messages import (
    BlobMeta,
    CommandAckMsg,
    FaultEventMsg,
    HeartbeatMsg,
    LinkStateMsg,
    utc_now_iso,
)
from flight.libs.types import AckStatus, FaultCode, LinkState, MessageType


def test_heartbeat_is_frozen() -> None:
    """Message dataclasses are immutable (frozen)."""
    hb = HeartbeatMsg(
        msg_type=MessageType.HEARTBEAT,
        timestamp_utc=utc_now_iso(),
        subsystem="payload",
        sequence=1,
    )
    try:
        # setattr (not direct assignment) so mypy does not statically reject writing
        # the read-only field; frozen dataclasses raise FrozenInstanceError (an
        # AttributeError subclass) at runtime.
        setattr(hb, "sequence", 2)  # noqa: B010
    except AttributeError:
        return
    raise AssertionError("HeartbeatMsg should be frozen")


def test_fault_event_carries_code() -> None:
    """FaultEventMsg carries a FaultCode and subsystem."""
    msg = FaultEventMsg(
        msg_type=MessageType.FAULT_EVENT,
        timestamp_utc=utc_now_iso(),
        fault_code=FaultCode.MODEL_CORRUPT,
        subsystem="payload",
        detail="checksum mismatch",
    )
    assert msg.fault_code is FaultCode.MODEL_CORRUPT


def test_blobmeta_constructs() -> None:
    """BlobMeta holds detection geometry."""
    blob = BlobMeta(
        blob_id=1,
        bbox=(10, 10, 20, 20),
        centroid_raw=(15.0, 15.0),
        pixel_area=100,
        mean_confidence=0.9,
        persistence_count=1,
    )
    assert blob.blob_id == 1


def test_utc_now_iso_format() -> None:
    """utc_now_iso returns an ISO 8601 UTC timestamp ending in Z."""
    stamp = utc_now_iso()
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$", stamp)


def test_raw_frame_msg_removed() -> None:
    """Frames never ride the bus: RawFrameMsg and RAW_FRAME no longer exist."""
    assert not hasattr(messages, "RawFrameMsg")
    assert not hasattr(MessageType, "RAW_FRAME")


def test_command_ack_msg_fields() -> None:
    """CommandAckMsg carries the ack status and command correlation handles."""
    ack = CommandAckMsg(
        msg_type=MessageType.COMMAND_ACK,
        timestamp_utc="2026-01-01T00:00:00.000Z",
        status=AckStatus.ACCEPTED,
        command_id="PING",
        source="ground",
        seq=1,
        fault_code=FaultCode.NONE,
        detail="",
    )
    assert ack.status is AckStatus.ACCEPTED
    assert ack.command_id == "PING"
    assert ack.seq == 1


def test_link_state_msg_fields() -> None:
    """LinkStateMsg carries the current AOS/LOS state."""
    msg = LinkStateMsg(
        msg_type=MessageType.LINK_STATE,
        timestamp_utc="2026-01-01T00:00:00.000Z",
        state=LinkState.AOS,
    )
    assert msg.state is LinkState.AOS
