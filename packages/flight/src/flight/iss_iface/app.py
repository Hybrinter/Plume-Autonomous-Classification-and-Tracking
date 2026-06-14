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
import base64
import binascii
import hashlib
import threading
from dataclasses import dataclass, field, replace

# internal
from flight.hal.interfaces import StationLink, StorageReader, StorageWriter
from flight.iss_iface.ingress import IngressOutcome, process_inbound
from flight.iss_iface.upload import ModelUploadState, add_chunk
from flight.libs.bus import MessageBus, Subscription
from flight.libs.ccsds import CcsdsHeader, encode_packet
from flight.libs.config import CommandIngressConfig, FaultConfig, LinkConfig, PactConfig
from flight.libs.messages import (
    CommandAckMsg,
    DownlinkItemMsg,
    FaultEventMsg,
    HeartbeatMsg,
    LinkStateMsg,
    ModelStagedMsg,
    RoutedCommandMsg,
)
from flight.libs.time import Clock
from flight.libs.types import (
    AckStatus,
    DownlinkPriority,
    Err,
    FaultCode,
    LinkState,
    MessageType,
    Ok,
)

HEARTBEAT_SUBSYSTEM = "iss_iface"
_UPLOAD_MODEL_CHUNK = "UPLOAD_MODEL_CHUNK"


@dataclass(slots=True)
class IngressState:
    """Mutable ingress state owned by the app shell (threaded through the pure core).

    Fields:
        last_seq: Per-source last-accepted sequence number map (replay guard).
        tm_sequence: Outbound TM packet sequence counter (wraps at 0x3FFF).
        upload: The in-progress chunked model-upload reassembly buffer.
    """

    last_seq: dict[str, int] = field(default_factory=dict)
    tm_sequence: int = 0
    upload: ModelUploadState = field(default_factory=ModelUploadState)


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
    storage_writer: StorageWriter
    bus: MessageBus
    clock: Clock
    downlink: Subscription[DownlinkItemMsg]
    routed: Subscription[RoutedCommandMsg]
    state: IngressState

    @staticmethod
    def from_config(
        cfg: PactConfig,
        bus: MessageBus,
        clock: Clock,
        link: StationLink,
        uplink_key: bytes,
        storage_reader: StorageReader,
        storage_writer: StorageWriter,
    ) -> IssIfaceApp:
        """Assemble an IssIfaceApp, subscribing to downlink items + routed model-upload chunks.

        Args:
            cfg: Top-level PactConfig (fault for timing; link + command_ingress for ingress).
            bus: The shared MessageBus to publish onto and subscribe from.
            clock: Injected Clock (real or manual).
            link: The injected StationLink driver (sim or real).
            uplink_key: The HMAC secret loaded by the composition root.
            storage_reader: The injected StorageReader for resolving product references at
                transmission time (the StorageService's read face).
            storage_writer: The injected StorageWriter for staging reassembled model artifacts
                (the StorageService's write face).

        Returns:
            An IssIfaceApp with fresh DownlinkItemMsg + RoutedCommandMsg subscriptions and empty
            ingress state.
        """
        return IssIfaceApp(
            fault_cfg=cfg.fault,
            link_cfg=cfg.link,
            ingress_cfg=cfg.command_ingress,
            uplink_key=uplink_key,
            link=link,
            storage_reader=storage_reader,
            storage_writer=storage_writer,
            bus=bus,
            clock=clock,
            downlink=bus.subscribe(DownlinkItemMsg),
            routed=bus.subscribe(RoutedCommandMsg),
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

    def pump_routed_commands(self) -> int:
        """Drain routed model-upload chunks; reassemble + stage a completed artifact.

        Returns:
            The number of UPLOAD_MODEL_CHUNK commands processed. Each chunk is base64-decoded
            and accumulated; on the final chunk the artifact is stored via the StorageWriter and
            a ModelStagedMsg is published for the core ModelDeployService. Each chunk gets an
            execution CommandAckMsg; a malformed/failed chunk acks REJECTED + emits a fault.
        """
        processed = 0
        while not self.routed.empty():
            command = self.routed.get_nowait()
            if command.target != HEARTBEAT_SUBSYSTEM or command.command_id != _UPLOAD_MODEL_CHUNK:
                continue
            processed += 1
            self._handle_chunk(command)
        return processed

    def _handle_chunk(self, command: RoutedCommandMsg) -> None:
        """Decode + accumulate one upload chunk; stage the artifact when reassembly completes."""
        try:
            data = base64.b64decode(str(command.params["data_b64"]), validate=True)
        except binascii.Error, ValueError:
            self._publish_fault(FaultCode.COMMAND_INVALID, "chunk data not valid base64")
            self._publish_exec_ack(
                command, AckStatus.REJECTED, FaultCode.COMMAND_INVALID, "bad b64"
            )
            return
        result = add_chunk(
            self.state.upload,
            int(command.params["chunk_index"]),
            int(command.params["total_chunks"]),
            data,
            int(command.params["crc32"]),
        )
        if result.fault is not None:
            self._publish_fault(result.fault, result.detail)
            self._publish_exec_ack(command, AckStatus.REJECTED, result.fault, result.detail)
            return
        if result.complete is None:
            self._publish_exec_ack(command, AckStatus.ACCEPTED, FaultCode.NONE, result.detail)
            return
        blob = result.complete
        stored = self.storage_writer.store("staged_model", blob, DownlinkPriority.SCIENCE_PRODUCT)
        if isinstance(stored, Err):
            self._publish_fault(stored.error, "staged model store failed")
            self._publish_exec_ack(command, AckStatus.REJECTED, stored.error, "store failed")
            return
        self.bus.publish(
            ModelStagedMsg(
                msg_type=MessageType.MODEL_STAGED,
                timestamp_utc=self.clock.wall_clock_iso(),
                entry_id=stored.value,
                sha256=hashlib.sha256(blob).hexdigest(),
                version="",
            )
        )
        self._publish_exec_ack(command, AckStatus.ACCEPTED, FaultCode.NONE, "model reassembled")

    def _publish_exec_ack(
        self, command: RoutedCommandMsg, status: AckStatus, fault: FaultCode, detail: str
    ) -> None:
        """Publish an execution CommandAckMsg correlated to a routed command."""
        self.bus.publish(
            CommandAckMsg(
                msg_type=MessageType.COMMAND_ACK,
                timestamp_utc=self.clock.wall_clock_iso(),
                status=status,
                command_id=command.command_id,
                source=command.source,
                seq=command.seq,
                fault_code=fault,
                detail=detail,
            )
        )

    def tick(self) -> None:
        """Publish link state, pump inbound commands + routed chunks, then pump downlinks once."""
        self.bus.publish(
            LinkStateMsg(
                msg_type=MessageType.LINK_STATE,
                timestamp_utc=self.clock.wall_clock_iso(),
                state=self.link.link_state(),
            )
        )
        self.pump_uplink()
        self.pump_routed_commands()
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
