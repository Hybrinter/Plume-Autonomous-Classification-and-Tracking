"""Tests for the CommandMsg station-command envelope."""

from flight.libs.messages import CommandMsg
from flight.libs.types import MessageType


def test_command_msg_fields() -> None:
    """CommandMsg carries target, command_id, params, source, and seq."""
    cmd = CommandMsg(
        msg_type=MessageType.COMMAND,
        timestamp_utc="2026-06-01T00:00:00.000Z",
        target="payload",
        command_id="set_mode",
        params={"mode": "ACTIVE", "dwell_s": 30},
        source="ground",
        seq=7,
    )
    assert cmd.target == "payload"
    assert cmd.command_id == "set_mode"
    assert cmd.params["mode"] == "ACTIVE"
    assert cmd.seq == 7
    assert cmd.msg_type is MessageType.COMMAND


def test_command_type_value_mirrors_name() -> None:
    """The new MessageType.COMMAND value mirrors its member name."""
    assert MessageType.COMMAND.value == "COMMAND"
