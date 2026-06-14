"""DownlinkManager tests: priority ordering, AOS gating, byte budget, product refs."""

import dataclasses

from flight.core.downlink import DownlinkManager
from flight.libs.bus import MessageBus
from flight.libs.config import PactConfig
from flight.libs.messages import (
    CommandAckMsg,
    DownlinkItemMsg,
    FaultEventMsg,
    LinkStateMsg,
    ProductRefMsg,
    TelemetryEventMsg,
)
from flight.libs.time import ManualClock
from flight.libs.types import (
    AckStatus,
    DownlinkPriority,
    FaultCode,
    LinkState,
    MessageType,
)


def _manager(bus: MessageBus, budget: int = 1_000_000) -> DownlinkManager:
    """Build a DownlinkManager with the given per-pass byte budget."""
    cfg = PactConfig()
    cfg = dataclasses.replace(
        cfg, comms=dataclasses.replace(cfg.comms, downlink_max_bytes_per_pass=budget)
    )
    return DownlinkManager.from_config(cfg, bus, ManualClock())


def _aos(bus: MessageBus) -> None:
    """Publish an AOS LinkStateMsg."""
    bus.publish(
        LinkStateMsg(msg_type=MessageType.LINK_STATE, timestamp_utc="t", state=LinkState.AOS)
    )


def _fault(bus: MessageBus) -> None:
    """Publish a fault event."""
    bus.publish(
        FaultEventMsg(
            msg_type=MessageType.FAULT_EVENT,
            timestamp_utc="t",
            fault_code=FaultCode.THERMAL_OVER_LIMIT,
            subsystem="thermal",
            detail="hot",
        )
    )


def _telemetry(bus: MessageBus) -> None:
    """Publish a housekeeping telemetry event."""
    bus.publish(
        TelemetryEventMsg(
            msg_type=MessageType.TELEMETRY_EVENT,
            timestamp_utc="t",
            subsystem="thermal",
            event_name="thermal_sample",
            payload={"temperature_c": 25.0},
        )
    )


def _drain(sub: object) -> list[DownlinkItemMsg]:
    """Drain a subscription into a list."""
    out: list[DownlinkItemMsg] = []
    while not sub.empty():  # type: ignore[attr-defined]
        out.append(sub.get_nowait())  # type: ignore[attr-defined]
    return out


def test_holds_during_los() -> None:
    """With no AOS, nothing is emitted (items wait in the queue)."""
    bus = MessageBus()
    mgr = _manager(bus)
    items = bus.subscribe(DownlinkItemMsg)
    _fault(bus)
    mgr.tick()
    assert items.empty()


def test_emits_in_priority_order_during_aos() -> None:
    """Fault > ack > HK telemetry ordering is honored in the emitted stream."""
    bus = MessageBus()
    mgr = _manager(bus)
    items = bus.subscribe(DownlinkItemMsg)
    _telemetry(bus)  # lowest of the three
    bus.publish(
        CommandAckMsg(
            msg_type=MessageType.COMMAND_ACK,
            timestamp_utc="t",
            status=AckStatus.ACCEPTED,
            command_id="PING",
            source="ground",
            seq=1,
            fault_code=FaultCode.NONE,
            detail="",
        )
    )
    _fault(bus)  # highest
    _aos(bus)
    mgr.tick()
    emitted = _drain(items)
    priorities = [i.priority for i in emitted]
    assert priorities == [
        DownlinkPriority.FAULT_EVENT,
        DownlinkPriority.COMMAND_ACK,
        DownlinkPriority.HK_TELEMETRY,
    ]


def test_byte_budget_defers_lower_priority_items() -> None:
    """A tight budget emits the highest-priority item and defers the rest to the next pass."""
    bus = MessageBus()
    mgr = _manager(bus, budget=1)  # only the first (highest-priority) item fits
    items = bus.subscribe(DownlinkItemMsg)
    _telemetry(bus)
    _fault(bus)
    _aos(bus)
    mgr.tick()
    first = _drain(items)
    assert len(first) == 1
    assert first[0].priority is DownlinkPriority.FAULT_EVENT
    # Next AOS pass drains the deferred telemetry item.
    _aos(bus)
    mgr.tick()
    second = _drain(items)
    assert len(second) == 1
    assert second[0].priority is DownlinkPriority.HK_TELEMETRY


def test_product_ref_becomes_storage_ref_item() -> None:
    """A ProductRefMsg is emitted as a storage-ref DownlinkItemMsg (no inline bytes)."""
    bus = MessageBus()
    mgr = _manager(bus)
    items = bus.subscribe(DownlinkItemMsg)
    bus.publish(
        ProductRefMsg(
            msg_type=MessageType.PRODUCT_REF,
            timestamp_utc="t",
            entry_id="00000001_mask",
            priority=DownlinkPriority.SCIENCE_PRODUCT,
            item_id="mask_thumb_1",
            byte_len=512,
        )
    )
    _aos(bus)
    mgr.tick()
    emitted = _drain(items)
    assert len(emitted) == 1
    assert emitted[0].storage_ref == "00000001_mask"
    assert emitted[0].payload_bytes == b""
