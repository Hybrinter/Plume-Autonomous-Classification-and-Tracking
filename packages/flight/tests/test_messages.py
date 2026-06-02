"""Tests for the migrated message dataclasses."""

import re

from flight.libs.messages import (
    BlobMeta,
    FaultEventMsg,
    HeartbeatMsg,
    utc_now_iso,
)
from flight.libs.types import FaultCode, MessageType


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
