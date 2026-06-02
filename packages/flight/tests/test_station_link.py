"""Conformance + behavior tests for the StationLink HAL and its drivers."""

from flight.hal.drivers_real import RealStationLink
from flight.hal.drivers_sim import SimStationLink
from flight.hal.interfaces import StationLink
from flight.libs.messages import CommandMsg, DownlinkItemMsg
from flight.libs.types import DownlinkPriority, MessageType, Ok


def _command(seq: int) -> CommandMsg:
    """Build a CommandMsg targeting the payload subsystem."""
    return CommandMsg(
        msg_type=MessageType.COMMAND,
        timestamp_utc="t",
        target="payload",
        command_id="set_mode",
        params={"mode": "ACTIVE"},
        source="ground",
        seq=seq,
    )


def _downlink_item() -> DownlinkItemMsg:
    """Build a minimal DownlinkItemMsg."""
    return DownlinkItemMsg(
        msg_type=MessageType.DOWNLINK_ITEM,
        timestamp_utc="t",
        priority=DownlinkPriority.HEALTH_TELEMETRY,
        payload_bytes=b"hello",
        crc32=0,
        item_id="item-1",
    )


def test_sim_station_link_satisfies_protocol() -> None:
    """SimStationLink conforms to StationLink (typed + runtime)."""
    link: StationLink = SimStationLink([])
    assert isinstance(link, StationLink)


def test_real_station_link_satisfies_protocol() -> None:
    """RealStationLink conforms to StationLink and constructs without hardware."""
    link: StationLink = RealStationLink()
    assert isinstance(link, StationLink)


def test_sim_receives_scripted_commands_in_order_then_none() -> None:
    """receive_command yields each scripted command once, then Ok(None)."""
    link = SimStationLink([_command(1), _command(2)])
    first = link.receive_command()
    second = link.receive_command()
    third = link.receive_command()
    assert isinstance(first, Ok) and first.value is not None and first.value.seq == 1
    assert isinstance(second, Ok) and second.value is not None and second.value.seq == 2
    assert isinstance(third, Ok) and third.value is None


def test_sim_records_downlinked_items() -> None:
    """send_downlink records each item; downlinked exposes them in order."""
    link = SimStationLink([])
    result = link.send_downlink(_downlink_item())
    assert isinstance(result, Ok)
    assert len(link.downlinked) == 1
    assert link.downlinked[0].item_id == "item-1"


def test_real_station_link_stub_is_inert() -> None:
    """RealStationLink stub returns no pending command and accepts downlinks."""
    link = RealStationLink()
    received = link.receive_command()
    assert isinstance(received, Ok) and received.value is None
    assert isinstance(link.send_downlink(_downlink_item()), Ok)
