"""ISS interface app: the authenticated command-ingress front door + downlink egress.

Inbound: receive_packet (raw CCSDS bytes) -> process_inbound (decode/CRC/HMAC/parse/validate/
dedup) -> publish CommandMsg (validated) + always publish a CommandAckMsg (ACCEPTED/REJECTED).
Outbound: when AOS, drain the DownlinkItemMsg stream the core downlink manager emits (the sole
prioritizer), encode each into a CCSDS TM packet, and send_packet. A DownlinkItemMsg carrying a
storage_ref is resolved to its bytes via the injected StorageReader at transmission time (the
large-artifact invariant); inline items send their payload_bytes directly. Each tick also
publishes the current LinkStateMsg. The ingress decision logic is a pure core
(flight.iss_iface.ingress); this shell owns the bus, clock, HMAC key, and the mutable sequence
state. Command acks are NOT downlinked here anymore -- they flow ingress -> bus -> downlink
manager -> DownlinkItemMsg so all downlink classes share one prioritized, budgeted path.

Contains:
  - IngressState: mutable per-source last-seq map + outbound TM sequence counter.
  - IssIfaceApp: from_config(); pump_uplink(); pump_downlink(); tick(); run(); helpers.

Satisfies: REQ-COMM-HIGH-001, REQ-COMM-HIGH-003, REQ-COMM-HIGH-004.
"""

from __future__ import annotations

# stdlib
import threading
from dataclasses import dataclass, field, replace

# internal
from flight.hal.interfaces import StationLink, StorageReader
from flight.iss_iface.ingress import IngressOutcome, process_inbound
from flight.libs.bus import MessageBus, Subscription
from flight.libs.ccsds import CcsdsHeader, encode_packet
from flight.libs.config import CommandIngressConfig, FaultConfig, LinkConfig, PactConfig
from flight.libs.messages import (
    CommandAckMsg,
    DownlinkItemMsg,
    FaultEventMsg,
    HeartbeatMsg,
    LinkStateMsg,
)
from flight.libs.time import Clock
from flight.libs.types import Err, FaultCode, LinkState, MessageType, Ok

HEARTBEAT_SUBSYSTEM = "iss_iface"


@dataclass(slots=True)
class IngressState:
    """Mutable ingress state owned by the app shell (threaded through the pure core).

    Fields:
        last_seq: Per-source last-accepted sequence number map (replay guard).
        tm_sequence: Outbound TM packet sequence counter (wraps at 0x3FFF).
    """

    last_seq: dict[str, int] = field(default_factory=dict)
    tm_sequence: int = 0


@dataclass(frozen=True)
class IssIfaceApp:
    """Station <-> bus command-ingress front door + downlink egress.

    Holds the injected link, bus, clock, config, HMAC key, subscriptions, and mutable
    IngressState. Constructed via from_config(); run() drives the periodic ingress loop.
    """

    fault_cfg: FaultConfig
    link_cfg: LinkConfig
    ingress_cfg: CommandIngressConfig
    uplink_key: bytes
    link: StationLink
    storage_reader: StorageReader
    bus: MessageBus
    clock: Clock
    downlink: Subscription[DownlinkItemMsg]
    state: IngressState

    @staticmethod
    def from_config(
        cfg: PactConfig,
        bus: MessageBus,
        clock: Clock,
        link: StationLink,
        uplink_key: bytes,
        storage_reader: StorageReader,
    ) -> IssIfaceApp:
        """Assemble an IssIfaceApp, subscribing to the downlink-manager item stream.

        Args:
            cfg: Top-level PactConfig (fault for timing; link + command_ingress for ingress).
            bus: The shared MessageBus to publish onto and subscribe from.
            clock: Injected Clock (real or manual).
            link: The injected StationLink driver (sim or real).
            uplink_key: The HMAC secret loaded by the composition root.
            storage_reader: The injected StorageReader for resolving product references at
                transmission time (the StorageService's read face).

        Returns:
            An IssIfaceApp with a fresh DownlinkItemMsg subscription and empty ingress state.
        """
        return IssIfaceApp(
            fault_cfg=cfg.fault,
            link_cfg=cfg.link,
            ingress_cfg=cfg.command_ingress,
            uplink_key=uplink_key,
            link=link,
            storage_reader=storage_reader,
            bus=bus,
            clock=clock,
            downlink=bus.subscribe(DownlinkItemMsg),
            state=IngressState(),
        )

    def pump_uplink(self) -> int:
        """Drain inbound packets; publish validated CommandMsgs; always ack each packet.

        Returns:
            The number of CommandMsg published (accepted commands). Each inbound packet --
            accepted or rejected -- produces exactly one CommandAckMsg; rejects also emit a
            FaultEventMsg. A link Err stops the drain early (preserves ordering).
        """
        published = 0
        while True:
            result = self.link.receive_packet()
            if isinstance(result, Err):
                self._publish_fault(result.error, "station uplink receive failed")
                break
            raw = result.value
            if raw is None:
                break
            outcome, self.state.last_seq = process_inbound(
                raw,
                self.uplink_key,
                self.ingress_cfg.require_auth,
                self.ingress_cfg.accepted_sources,
                self.state.last_seq,
            )
            if outcome.command is not None:
                stamped = replace(outcome.command, timestamp_utc=self.clock.wall_clock_iso())
                self.bus.publish(stamped)
                published += 1
            else:
                self._publish_fault(outcome.fault_code, outcome.detail)
            self._publish_ack(outcome)
        return published

    def pump_downlink(self) -> int:
        """When AOS, encode and send the downlink manager's DownlinkItemMsg stream as TM packets.

        Returns:
            The number of packets sent. During LOS nothing is drained (items wait in the
            subscription queue). A storage_ref item is resolved to its bytes via the injected
            StorageReader; a failed resolution emits a fault and is skipped. A send Err emits a
            fault and is not counted.
        """
        if self.link.link_state() is not LinkState.AOS:
            return 0
        sent = 0
        while not self.downlink.empty():
            item = self.downlink.get_nowait()
            if item.storage_ref:
                resolved = self.storage_reader.read(item.storage_ref)
                if isinstance(resolved, Err):
                    self._publish_fault(resolved.error, f"downlink read failed {item.storage_ref}")
                    continue
                body = resolved.value
            else:
                body = item.payload_bytes
            sent += self._send_tm(body)
        return sent

    def tick(self) -> None:
        """Publish link state, pump inbound commands, then pump outbound downlinks once."""
        self.bus.publish(
            LinkStateMsg(
                msg_type=MessageType.LINK_STATE,
                timestamp_utc=self.clock.wall_clock_iso(),
                state=self.link.link_state(),
            )
        )
        self.pump_uplink()
        self.pump_downlink()

    def run(self, stop_event: threading.Event) -> None:
        """Run the ingress loop until stop_event is set, emitting periodic heartbeats.

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
                        subsystem=HEARTBEAT_SUBSYSTEM,
                        sequence=sequence,
                    )
                )
                sequence += 1
                last_heartbeat = now
            stop_event.wait(timeout=self.fault_cfg.watchdog_interval_s)
        self.link.close()

    def _send_tm(self, body: bytes) -> int:
        """Encode body into a CCSDS TM packet and send it; return 1 on success, 0 on error."""
        if len(body) == 0:
            return 0
        encoded = encode_packet(
            CcsdsHeader(
                packet_type=0,
                apid=self.link_cfg.tm_apid,
                sequence_count=self.state.tm_sequence & 0x3FFF,
            ),
            body,
        )
        if isinstance(encoded, Err):
            self._publish_fault(encoded.error, "tm encode failed")
            return 0
        self.state.tm_sequence += 1
        result = self.link.send_packet(encoded.value)
        if isinstance(result, Ok):
            return 1
        self._publish_fault(result.error, "station downlink send failed")
        return 0

    def _publish_ack(self, outcome: IngressOutcome) -> None:
        """Publish a CommandAckMsg for one ingress outcome (always, accept or reject)."""
        self.bus.publish(
            CommandAckMsg(
                msg_type=MessageType.COMMAND_ACK,
                timestamp_utc=self.clock.wall_clock_iso(),
                status=outcome.status,
                command_id=outcome.command_id,
                source=outcome.source,
                seq=outcome.seq,
                fault_code=outcome.fault_code,
                detail=outcome.detail,
            )
        )

    def _publish_fault(self, code: FaultCode, detail: str) -> None:
        """Publish a FaultEventMsg from the iss_iface subsystem onto the bus."""
        self.bus.publish(
            FaultEventMsg(
                msg_type=MessageType.FAULT_EVENT,
                timestamp_utc=self.clock.wall_clock_iso(),
                fault_code=code,
                subsystem=HEARTBEAT_SUBSYSTEM,
                detail=detail,
            )
        )
