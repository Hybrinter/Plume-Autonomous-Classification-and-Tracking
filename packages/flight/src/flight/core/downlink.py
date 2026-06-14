"""Core-hosted downlink manager: the single prioritized, AOS-gated, budgeted downlink path.

The downlink manager is the sole producer of DownlinkItemMsg (spec Section 6). It subscribes to
every downlinkable class, enqueues each by priority -- fault events > command acks > housekeeping
telemetry > science products -- and, only while the link is AOS and within a per-pass byte budget,
emits DownlinkItemMsg in priority order for iss_iface to transmit. Compact items (faults, acks,
telemetry) are serialized inline; science products are carried as a storage reference (entry id)
that iss_iface resolves via the injected StorageReader at transmission time, keeping large
artifacts off the bus.

AOS/LOS is read from the LinkStateMsg iss_iface publishes each tick; during LOS nothing is
emitted (items wait in the queue). The byte budget bounds one pass so a backlog drains across
passes rather than flooding a contact; the highest-priority item is always allowed through even
if it alone exceeds the budget (no starvation).

Contains:
  - _QueuedItem / DownlinkState: the priority queue + AOS flag threaded as mutable shell state.
  - DownlinkManager: from_config(); tick(); run().

Satisfies: REQ-COMM-HIGH-001, REQ-DATA-DOWNLINK-001.
"""

from __future__ import annotations

# stdlib
import json
import threading
import zlib
from dataclasses import dataclass, field

# internal
from flight.libs.bus import MessageBus, Subscription
from flight.libs.config import CommsConfig, FaultConfig, PactConfig
from flight.libs.messages import (
    CommandAckMsg,
    DownlinkItemMsg,
    FaultEventMsg,
    HeartbeatMsg,
    LinkStateMsg,
    ProductRefMsg,
    TelemetryEventMsg,
)
from flight.libs.time import Clock
from flight.libs.types import DownlinkPriority, LinkState, MessageType

SUBSYSTEM = "downlink"


@dataclass(slots=True, frozen=True)
class _QueuedItem:
    """One pending downlink item: inline bytes or a storage reference, with priority + order."""

    priority: DownlinkPriority
    order: int
    item_id: str
    payload_bytes: bytes  # inline content ("" when storage_ref set)
    storage_ref: str  # storage entry id ("" when inline)
    byte_len: int  # size used for budget accounting
    crc32: int  # CRC-32 of payload_bytes (0 for storage_ref items)


@dataclass(slots=True)
class DownlinkState:
    """Mutable downlink state owned by the shell.

    Fields:
        pending: The unsent queued items (drained in priority order each AOS pass).
        next_order: Monotonic enqueue counter (oldest-first tie-break within a priority).
        aos: Latest link-acquisition state (True == AOS, drain enabled).
    """

    pending: list[_QueuedItem] = field(default_factory=list)
    next_order: int = 0
    aos: bool = False


@dataclass(frozen=True)
class DownlinkManager:
    """Prioritized AOS-gated downlink manager: bus events -> ordered DownlinkItemMsg stream."""

    cfg: CommsConfig
    fault_cfg: FaultConfig
    bus: MessageBus
    clock: Clock
    faults: Subscription[FaultEventMsg]
    acks: Subscription[CommandAckMsg]
    telemetry: Subscription[TelemetryEventMsg]
    products: Subscription[ProductRefMsg]
    link: Subscription[LinkStateMsg]
    state: DownlinkState

    @staticmethod
    def from_config(cfg: PactConfig, bus: MessageBus, clock: Clock) -> DownlinkManager:
        """Assemble a DownlinkManager subscribing to every downlinkable class + link state.

        Args:
            cfg: Top-level PactConfig (comms for the per-pass budget; fault for heartbeat).
            bus: The shared MessageBus to subscribe to / publish onto.
            clock: Injected Clock (real or manual).

        Returns:
            A DownlinkManager with fresh subscriptions and an empty (LOS) queue.
        """
        return DownlinkManager(
            cfg=cfg.comms,
            fault_cfg=cfg.fault,
            bus=bus,
            clock=clock,
            faults=bus.subscribe(FaultEventMsg),
            acks=bus.subscribe(CommandAckMsg),
            telemetry=bus.subscribe(TelemetryEventMsg),
            products=bus.subscribe(ProductRefMsg),
            link=bus.subscribe(LinkStateMsg),
            state=DownlinkState(),
        )

    def tick(self) -> None:
        """Drain inbound classes into the priority queue, then emit within AOS + the byte budget."""
        while not self.link.empty():
            self.state.aos = self.link.get_nowait().state is LinkState.AOS

        while not self.faults.empty():
            fault = self.faults.get_nowait()
            body = json.dumps(
                {
                    "type": "fault",
                    "ts": fault.timestamp_utc,
                    "fault_code": fault.fault_code.value,
                    "subsystem": fault.subsystem,
                    "detail": fault.detail,
                },
                separators=(",", ":"),
            ).encode("utf-8")
            self._enqueue_inline(
                DownlinkPriority.FAULT_EVENT, f"fault_{fault.fault_code.value}", body
            )
        while not self.acks.empty():
            ack = self.acks.get_nowait()
            body = json.dumps(
                {
                    "type": "command_ack",
                    "status": ack.status.value,
                    "command_id": ack.command_id,
                    "source": ack.source,
                    "seq": ack.seq,
                    "fault_code": ack.fault_code.value,
                    "detail": ack.detail,
                },
                separators=(",", ":"),
            ).encode("utf-8")
            self._enqueue_inline(DownlinkPriority.COMMAND_ACK, f"ack_{ack.source}_{ack.seq}", body)
        while not self.telemetry.empty():
            event = self.telemetry.get_nowait()
            body = json.dumps(
                {
                    "type": "telemetry",
                    "ts": event.timestamp_utc,
                    "subsystem": event.subsystem,
                    "event": event.event_name,
                    "payload": event.payload,
                },
                separators=(",", ":"),
            ).encode("utf-8")
            self._enqueue_inline(DownlinkPriority.HK_TELEMETRY, f"telem_{event.event_name}", body)
        while not self.products.empty():
            product = self.products.get_nowait()
            self.state.pending.append(
                _QueuedItem(
                    priority=product.priority,
                    order=self.state.next_order,
                    item_id=product.item_id,
                    payload_bytes=b"",
                    storage_ref=product.entry_id,
                    byte_len=product.byte_len,
                    crc32=0,
                )
            )
            self.state.next_order += 1

        if not self.state.aos:
            return
        self._drain_within_budget()

    def _enqueue_inline(self, priority: DownlinkPriority, item_id: str, body: bytes) -> None:
        """Append an inline (compact) item to the pending queue."""
        self.state.pending.append(
            _QueuedItem(
                priority=priority,
                order=self.state.next_order,
                item_id=item_id,
                payload_bytes=body,
                storage_ref="",
                byte_len=len(body),
                crc32=zlib.crc32(body) & 0xFFFFFFFF,
            )
        )
        self.state.next_order += 1

    def _drain_within_budget(self) -> None:
        """Emit pending items in priority order until the per-pass byte budget is exhausted."""
        budget = self.cfg.downlink_max_bytes_per_pass
        ordered = sorted(self.state.pending, key=lambda i: (i.priority.value, i.order))
        sent_bytes = 0
        remaining: list[_QueuedItem] = []
        for item in ordered:
            if sent_bytes == 0 or sent_bytes + item.byte_len <= budget:
                self.bus.publish(
                    DownlinkItemMsg(
                        msg_type=MessageType.DOWNLINK_ITEM,
                        timestamp_utc=self.clock.wall_clock_iso(),
                        priority=item.priority,
                        payload_bytes=item.payload_bytes,
                        crc32=item.crc32,
                        item_id=item.item_id,
                        storage_ref=item.storage_ref,
                    )
                )
                sent_bytes += item.byte_len
            else:
                remaining.append(item)
        self.state.pending = remaining

    def run(self, stop_event: threading.Event) -> None:
        """Run the downlink loop until stop_event is set, emitting periodic heartbeats.

        Args:
            stop_event: threading.Event; the loop exits cleanly once it is set.
        """
        sequence = 0
        last_heartbeat = self.clock.monotonic_s()
        while not stop_event.is_set():
            self.tick()
            now = self.clock.monotonic_s()
            if now - last_heartbeat >= self.fault_cfg.watchdog_interval_s:
                self.bus.publish(
                    HeartbeatMsg(
                        msg_type=MessageType.HEARTBEAT,
                        timestamp_utc=self.clock.wall_clock_iso(),
                        subsystem=SUBSYSTEM,
                        sequence=sequence,
                    )
                )
                sequence += 1
                last_heartbeat = now
            stop_event.wait(timeout=self.fault_cfg.watchdog_interval_s)
