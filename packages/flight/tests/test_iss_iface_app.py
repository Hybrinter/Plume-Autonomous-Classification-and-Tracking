"""Integration tests for the iss_iface app (station <-> bus bridge)."""

from flight.hal.drivers_sim import SimStationLink
from flight.iss_iface.app import IssIfaceApp
from flight.libs.bus import MessageBus
from flight.libs.config import PactConfig
from flight.libs.messages import CommandMsg, DownlinkItemMsg
from flight.libs.time import ManualClock
from flight.libs.types import DownlinkPriority, MessageType


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


def _downlink_item(item_id: str) -> DownlinkItemMsg:
    """Build a minimal DownlinkItemMsg with the given id."""
    return DownlinkItemMsg(
        msg_type=MessageType.DOWNLINK_ITEM,
        timestamp_utc="t",
        priority=DownlinkPriority.HEALTH_TELEMETRY,
        payload_bytes=b"x",
        crc32=0,
        item_id=item_id,
    )


def _app(inbound: list[CommandMsg]) -> tuple[IssIfaceApp, MessageBus, SimStationLink]:
    """Assemble an IssIfaceApp over a SimStationLink and a fresh bus."""
    bus = MessageBus()
    link = SimStationLink(inbound)
    app = IssIfaceApp.from_config(PactConfig(), bus, ManualClock(), link)
    return app, bus, link


def test_pump_uplink_publishes_commands_to_bus() -> None:
    """Inbound station commands are republished onto the bus as CommandMsg."""
    app, bus, _link = _app([_command(1), _command(2)])
    cmd_sub = bus.subscribe(CommandMsg)
    count = app.pump_uplink()
    assert count == 2
    received = []
    while not cmd_sub.empty():
        received.append(cmd_sub.get_nowait())
    assert [c.seq for c in received] == [1, 2]


def test_pump_downlink_forwards_bus_items_to_station() -> None:
    """DownlinkItemMsg published on the bus is forwarded to the station link."""
    app, bus, link = _app([])
    bus.publish(_downlink_item("a"))
    bus.publish(_downlink_item("b"))
    count = app.pump_downlink()
    assert count == 2
    assert [item.item_id for item in link.downlinked] == ["a", "b"]


def test_tick_pumps_both_directions() -> None:
    """tick() pumps inbound commands and outbound downlinks in one call."""
    app, bus, link = _app([_command(5)])
    cmd_sub = bus.subscribe(CommandMsg)
    bus.publish(_downlink_item("z"))
    app.tick()
    assert not cmd_sub.empty()
    assert cmd_sub.get_nowait().seq == 5
    assert [item.item_id for item in link.downlinked] == ["z"]
