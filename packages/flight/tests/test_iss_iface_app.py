"""Integration tests for the iss_iface app (authenticated command-ingress front door).

Verifies that the IssIfaceApp shell correctly: accepts a signed packet (publishes CommandMsg
with dictionary-stamped target + CommandAckMsg ACCEPTED); rejects a tampered packet (zero
CommandMsg + CommandAckMsg REJECTED with COMMAND_AUTH_FAIL); and gates downlink on AOS/LOS
(acks held in the subscription queue during LOS, drained under AOS).
"""

from flight.hal.drivers_sim import SimStationLink
from flight.iss_iface.app import IssIfaceApp
from flight.libs.bus import MessageBus
from flight.libs.commands import build_tc_packet
from flight.libs.config import PactConfig
from flight.libs.messages import CommandAckMsg, CommandMsg, DownlinkItemMsg, FaultEventMsg
from flight.libs.time import ManualClock
from flight.libs.types import (
    AckStatus,
    DownlinkPriority,
    Err,
    FaultCode,
    LinkState,
    MessageType,
    Ok,
    Result,
)

_KEY = b"test-iss-iface-key-00000000000000"


class _MemStorageReader:
    """In-memory StorageReader double for iss_iface tests (entry id -> bytes)."""

    def __init__(self, items: dict[str, bytes] | None = None) -> None:
        """Start with an optional pre-populated entry map."""
        self.items = items or {}

    def read(self, entry_id: str) -> Result[bytes, FaultCode]:
        """Return the stored bytes for entry_id, or Err(STORAGE_CORRUPT) when missing."""
        if entry_id in self.items:
            return Ok(self.items[entry_id])
        return Err(FaultCode.STORAGE_CORRUPT)


def _app_with_link(
    link: SimStationLink, storage: _MemStorageReader | None = None
) -> tuple[IssIfaceApp, MessageBus]:
    """Build an IssIfaceApp over the given link with a fresh bus and the shared test key."""
    bus = MessageBus()
    app = IssIfaceApp.from_config(
        PactConfig(), bus, ManualClock(), link, _KEY, storage or _MemStorageReader()
    )
    return app, bus


def _downlink_item(payload: bytes = b"x", storage_ref: str = "") -> DownlinkItemMsg:
    """Build a DownlinkItemMsg (inline or storage-ref) for downlink-egress tests."""
    return DownlinkItemMsg(
        msg_type=MessageType.DOWNLINK_ITEM,
        timestamp_utc="2026-01-01T00:00:00.000Z",
        priority=DownlinkPriority.COMMAND_ACK,
        payload_bytes=payload,
        crc32=0,
        item_id="item-1",
        storage_ref=storage_ref,
    )


def test_valid_signed_command_produces_command_msg_and_accepted_ack() -> None:
    """A correctly signed packet publishes CommandMsg (target stamped) + ACCEPTED ack."""
    pkt = build_tc_packet("SET_THERMAL_LIMIT", {"limit_c": 70.0}, "ground", 1, _KEY, apid=1)
    link = SimStationLink([pkt])
    app, bus = _app_with_link(link)
    cmd_sub = bus.subscribe(CommandMsg)
    ack_sub = bus.subscribe(CommandAckMsg)

    app.tick()

    cmds = []
    while not cmd_sub.empty():
        cmds.append(cmd_sub.get_nowait())
    acks = []
    while not ack_sub.empty():
        acks.append(ack_sub.get_nowait())

    assert len(cmds) == 1
    assert cmds[0].target == "thermal"
    assert cmds[0].command_id == "SET_THERMAL_LIMIT"
    assert len(acks) >= 1
    ingress_acks = [a for a in acks if a.status is AckStatus.ACCEPTED]
    assert len(ingress_acks) == 1


def test_tampered_packet_produces_no_command_msg_and_rejected_ack() -> None:
    """A packet signed with the wrong key yields zero CommandMsg + REJECTED ack."""
    pkt = build_tc_packet("PING", {}, "ground", 1, b"wrong-key-for-tamper-test-000000", apid=1)
    link = SimStationLink([pkt])
    app, bus = _app_with_link(link)
    cmd_sub = bus.subscribe(CommandMsg)
    ack_sub = bus.subscribe(CommandAckMsg)

    app.tick()

    cmds = []
    while not cmd_sub.empty():
        cmds.append(cmd_sub.get_nowait())
    acks = []
    while not ack_sub.empty():
        acks.append(ack_sub.get_nowait())

    assert len(cmds) == 0
    rejected = [a for a in acks if a.status is AckStatus.REJECTED]
    assert len(rejected) == 1
    assert rejected[0].fault_code is FaultCode.COMMAND_AUTH_FAIL


def test_pump_downlink_holds_items_during_los() -> None:
    """DownlinkItemMsg queued during LOS are held in the subscription and not sent."""
    link = SimStationLink(link_state=LinkState.LOS)
    app, bus = _app_with_link(link)
    bus.publish(_downlink_item())
    sent = app.pump_downlink()
    assert sent == 0
    assert len(link.sent) == 0


def test_pump_downlink_drains_items_during_aos() -> None:
    """Inline DownlinkItemMsg are encoded and sent as TM packets when the link is AOS."""
    link = SimStationLink(link_state=LinkState.AOS)
    app, bus = _app_with_link(link)
    bus.publish(_downlink_item())
    sent = app.pump_downlink()
    assert sent == 1
    assert len(link.sent) == 1


def test_pump_downlink_resolves_storage_ref() -> None:
    """A storage-ref DownlinkItemMsg is resolved to its bytes via the StorageReader and sent."""
    link = SimStationLink(link_state=LinkState.AOS)
    app, bus = _app_with_link(link, _MemStorageReader({"entry-1": b"product-bytes"}))
    bus.publish(_downlink_item(payload=b"", storage_ref="entry-1"))
    sent = app.pump_downlink()
    assert sent == 1
    assert len(link.sent) == 1


def test_pump_downlink_skips_unresolvable_storage_ref() -> None:
    """A storage-ref that cannot be resolved emits a fault and is not sent."""
    link = SimStationLink(link_state=LinkState.AOS)
    app, bus = _app_with_link(link, _MemStorageReader({}))
    fault_sub = bus.subscribe(FaultEventMsg)
    bus.publish(_downlink_item(payload=b"", storage_ref="missing"))
    sent = app.pump_downlink()
    assert sent == 0
    assert not fault_sub.empty()


def test_tick_publishes_link_state_msg() -> None:
    """tick() always publishes a LinkStateMsg reflecting the current AOS/LOS state."""
    from flight.libs.messages import LinkStateMsg

    link = SimStationLink(link_state=LinkState.AOS)
    app, bus = _app_with_link(link)
    ls_sub = bus.subscribe(LinkStateMsg)

    app.tick()

    assert not ls_sub.empty()
    msg = ls_sub.get_nowait()
    assert msg.state is LinkState.AOS
