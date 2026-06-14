"""Every bus envelope carries the schema_version field (spec Section 7)."""

import dataclasses

from flight.libs import messages as m
from flight.libs.messages import SCHEMA_VERSION, HeartbeatMsg
from flight.libs.types import MessageType


def test_heartbeat_has_schema_version() -> None:
    """A constructed message defaults schema_version to the module constant."""
    hb = HeartbeatMsg(msg_type=MessageType.HEARTBEAT, timestamp_utc="t", subsystem="x", sequence=1)
    assert hb.schema_version == SCHEMA_VERSION


def test_every_message_dataclass_has_schema_version() -> None:
    """Every *Msg envelope dataclass declares a schema_version field."""
    msg_classes = [
        obj
        for name, obj in vars(m).items()
        if name.endswith("Msg") and isinstance(obj, type) and dataclasses.is_dataclass(obj)
    ]
    assert msg_classes  # sanity: we found the envelopes
    for cls in msg_classes:
        field_names = {f.name for f in dataclasses.fields(cls)}
        assert "schema_version" in field_names, f"{cls.__name__} missing schema_version"
