"""Telemetry reporter — formats health snapshots and events into CCSDS packets.

Provides:
  - format_health_packet()       — serialise SystemHealthSnapshot → CcsdsPacket
  - format_telemetry_event()     — serialise TelemetryEventMsg → CcsdsPacket
  - run_telemetry_process()      — threading.Thread entry point

Satisfies: REQ-OPER-HIGH-001, REQ-COMM-HIGH-001.
"""

from __future__ import annotations

# stdlib
import json
import queue
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

# internal
from pact.types.enums import DownlinkPriority, MessageType
from pact.types.messages import (
    DownlinkItemMsg,
    FaultEventMsg,
    HeartbeatMsg,
    TelemetryEventMsg,
)
from pact.telemetry.health import SystemHealthSnapshot

if TYPE_CHECKING:
    # CcsdsPacket lives in pact.comms.ccsds, which does not exist yet (Phase I stub).
    # Import only for type checking to avoid a circular / missing import at runtime.
    from pact.comms.ccsds import CcsdsPacket  # noqa: F401

import structlog

log = structlog.get_logger().bind(subsystem="telemetry")


# ---------------------------------------------------------------------------
# Packet formatters
# ---------------------------------------------------------------------------


def format_health_packet(snapshot: SystemHealthSnapshot, apid: int) -> "CcsdsPacket":
    """Serialise a SystemHealthSnapshot into a CCSDS telemetry packet.

    The snapshot is JSON-encoded and placed in the CCSDS data field.
    sequence_count is set to 0; the caller is responsible for incrementing it.

    # TODO: replace JSON encoding with a compact binary format (TLM database) when the
    #        ground segment defines the packet ICD.
    """
    # Lazy import so this module loads even when pact.comms.ccsds is not yet present.
    from pact.comms.ccsds import CcsdsPacket  # type: ignore[import]

    payload: dict[str, object] = {
        "timestamp_utc": snapshot.timestamp_utc,
        "system_mode": snapshot.system_mode.value,
        "gimbal_state": snapshot.gimbal_state.value,
        "active_faults": [f.value for f in snapshot.active_faults],
        "frames_captured_today": snapshot.frames_captured_today,
        "bytes_downlinked_today": snapshot.bytes_downlinked_today,
        "bytes_remaining_today": snapshot.bytes_remaining_today,
        "model_version": snapshot.model_version,
        "model_deploy_state": snapshot.model_deploy_state.value,
        "inference_latency_ms_mean": snapshot.inference_latency_ms_mean,
        "inference_latency_ms_max": snapshot.inference_latency_ms_max,
        "storage_bytes_used": snapshot.storage_bytes_used,
    }
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return CcsdsPacket(
        version=0,
        packet_type=0,         # telemetry
        sec_hdr_flag=0,
        apid=apid,
        sequence_flags=0b11,   # unsegmented
        sequence_count=0,      # caller should set the real counter
        data_length=len(data) - 1,
        data=data,
    )


def format_telemetry_event(event: TelemetryEventMsg, apid: int) -> "CcsdsPacket":
    """Serialise a TelemetryEventMsg into a CCSDS telemetry packet.

    # TODO: replace JSON encoding with a compact binary TLM format.
    """
    from pact.comms.ccsds import CcsdsPacket  # type: ignore[import]

    payload: dict[str, object] = {
        "timestamp_utc": event.timestamp_utc,
        "subsystem": event.subsystem,
        "event_name": event.event_name,
        "payload": event.payload,
    }
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return CcsdsPacket(
        version=0,
        packet_type=0,
        sec_hdr_flag=0,
        apid=apid,
        sequence_flags=0b11,
        sequence_count=0,
        data_length=len(data) - 1,
        data=data,
    )


# ---------------------------------------------------------------------------
# Process entry point
# ---------------------------------------------------------------------------


def run_telemetry_process(
    apid: int,
    telemetry_queue: "queue.Queue[TelemetryEventMsg]",
    downlink_queue: "queue.Queue[DownlinkItemMsg]",
    heartbeat_queue: "queue.Queue[HeartbeatMsg]",
    stop_event: "queue.Queue[None] | None" = None,
) -> None:
    """Telemetry threading.Thread entry point.

    Drains TelemetryEventMsg from telemetry_queue, formats each as a CCSDS packet,
    and enqueues a DownlinkItemMsg at DownlinkPriority.HEALTH_TELEMETRY.

    Heartbeats are emitted approximately every 5 seconds (approximated by the drain loop
    timeout).

    # TODO: build a rolling SystemHealthSnapshot from aggregated events and emit it
    #        periodically rather than per-event.
    # TODO: read heartbeat_interval_s from FaultConfig (pass config in as argument).
    # TODO: implement CRC-32 of the serialised packet payload before enqueueing.
    """
    HEARTBEAT_INTERVAL_S: float = 5.0
    last_heartbeat: float = time.monotonic()
    sequence: int = 0

    log.info("telemetry_process_started", apid=apid)

    while True:
        try:
            event: TelemetryEventMsg = telemetry_queue.get(timeout=1.0)
        except queue.Empty:
            _maybe_send_heartbeat(heartbeat_queue, last_heartbeat, HEARTBEAT_INTERVAL_S, sequence)
            continue

        now_str = datetime.now(timezone.utc).isoformat(timespec="milliseconds")

        # TODO: call format_telemetry_event() once comms.ccsds is implemented;
        #       for now, serialise directly to bytes to avoid import dependency.
        payload_bytes = json.dumps(
            {
                "timestamp_utc": event.timestamp_utc,
                "subsystem": event.subsystem,
                "event_name": event.event_name,
                "payload": event.payload,
            },
            separators=(",", ":"),
        ).encode("utf-8")

        downlink_item = DownlinkItemMsg(
            msg_type=MessageType.DOWNLINK_ITEM,
            timestamp_utc=now_str,
            priority=DownlinkPriority.HEALTH_TELEMETRY,
            payload_bytes=payload_bytes,
            crc32=0,                    # TODO: compute real CRC-32
            item_id=f"telemetry-{event.subsystem}-{event.event_name}-{now_str}",
        )
        try:
            downlink_queue.put_nowait(downlink_item)
        except queue.Full:
            log.warning("downlink_queue_full", subsystem=event.subsystem)

        now_mono = time.monotonic()
        if now_mono - last_heartbeat >= HEARTBEAT_INTERVAL_S:
            _emit_heartbeat(heartbeat_queue, sequence)
            last_heartbeat = now_mono
            sequence += 1

        log.debug("telemetry_event_processed", subsystem=event.subsystem, event=event.event_name)


def _emit_heartbeat(
    heartbeat_queue: "queue.Queue[HeartbeatMsg]",
    sequence: int,
) -> None:
    now_str = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    try:
        heartbeat_queue.put_nowait(
            HeartbeatMsg(
                msg_type=MessageType.HEARTBEAT,
                timestamp_utc=now_str,
                subsystem="telemetry",
                sequence=sequence,
            )
        )
    except queue.Full:
        log.warning("heartbeat_queue_full")


def _maybe_send_heartbeat(
    heartbeat_queue: "queue.Queue[HeartbeatMsg]",
    last_heartbeat: float,
    interval_s: float,
    sequence: int,
) -> None:
    now_mono = time.monotonic()
    if now_mono - last_heartbeat >= interval_s:
        _emit_heartbeat(heartbeat_queue, sequence)
