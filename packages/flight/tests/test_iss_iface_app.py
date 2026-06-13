"""Integration tests for the iss_iface app (station <-> bus bridge).

Note: during the Task 6 transition window, SimStationLink's legacy receive_command/
send_downlink methods are no-ops; pump_uplink returns 0 and pump_downlink drains items
without recording them on the link. These tests verify the app shell still wires and
runs cleanly. The full byte-level ingress tests are added in Task 7.
"""

from flight.hal.drivers_sim import SimStationLink
from flight.iss_iface.app import IssIfaceApp
from flight.libs.bus import MessageBus
from flight.libs.config import PactConfig
from flight.libs.messages import DownlinkItemMsg, FaultEventMsg
from flight.libs.time import ManualClock
from flight.libs.types import DownlinkPriority, MessageType


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


def _app() -> tuple[IssIfaceApp, MessageBus, SimStationLink]:
    """Assemble an IssIfaceApp over a SimStationLink and a fresh bus."""
    bus = MessageBus()
    link = SimStationLink()
    app = IssIfaceApp.from_config(PactConfig(), bus, ManualClock(), link)
    return app, bus, link


def test_pump_uplink_returns_zero_during_legacy_transition() -> None:
    """pump_uplink returns 0 while the link's legacy receive_command is a no-op."""
    app, _bus, _link = _app()
    assert app.pump_uplink() == 0


def test_pump_downlink_drains_bus_items() -> None:
    """DownlinkItemMsg on the bus is drained; count matches the number of items."""
    app, bus, _link = _app()
    bus.publish(_downlink_item("a"))
    bus.publish(_downlink_item("b"))
    count = app.pump_downlink()
    assert count == 2


def test_tick_runs_without_error() -> None:
    """tick() completes without raising even with no inbound commands."""
    app, bus, _link = _app()
    bus.publish(_downlink_item("z"))
    app.tick()  # should not raise


def test_no_fault_emitted_on_clean_tick() -> None:
    """A clean tick with no link errors produces no FaultEventMsg on the bus."""
    app, bus, _link = _app()
    fault_sub = bus.subscribe(FaultEventMsg)
    app.tick()
    assert fault_sub.empty()
