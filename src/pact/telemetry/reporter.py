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
import pickle
import queue
import time
import zlib
from datetime import datetime, timezone
from typing import TYPE_CHECKING

# internal
from pact.types.enums import (
    DownlinkPriority,
    GimbalState,
    MessageType,
    ModelDeployState,
    SystemMode,
)
from pact.types.messages import (
    DownlinkItemMsg,
    FaultEventMsg,
    HeartbeatMsg,
    TelemetryEventMsg,
    utc_now_iso,
)
from pact.telemetry.health import SystemHealthSnapshot

if TYPE_CHECKING:
    # CcsdsPacket lives in pact.comms.ccsds, which does not exist yet (Phase I stub).
    # Import only for type checking to avoid a circular / missing import at runtime.
    from pact.comms.ccsds import CcsdsPacket  # noqa: F401

import structlog

log = structlog.get_logger().bind(subsystem="telemetry")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_telemetry_ccsds_packet(apid: int, data: bytes) -> "CcsdsPacket":
    """Build an unsegmented CCSDS telemetry packet from a pre-serialized payload.

    Lazy-imports CcsdsPacket so this module loads even when pact.comms.ccsds is absent.
    sequence_count is set to 0; the caller is responsible for incrementing it.
    """
    from pact.comms.ccsds import CcsdsPacket  # type: ignore[import]

    return CcsdsPacket(
        version=0,
        packet_type=0,         # telemetry
        sec_hdr_flag=0,
        apid=apid,
        sequence_flags=0b11,   # unsegmented
        sequence_count=0,
        data_length=len(data) - 1,
        data=data,
    )


# ---------------------------------------------------------------------------
# Packet formatters
# ---------------------------------------------------------------------------


def format_health_packet(snapshot: SystemHealthSnapshot, apid: int) -> "CcsdsPacket":
    """Serialise a SystemHealthSnapshot into a CCSDS telemetry packet.

    The snapshot is JSON-encoded and placed in the CCSDS data field.

    # TODO: replace JSON encoding with a compact binary format (TLM database) when the
    #        ground segment defines the packet ICD.
    """
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
    return _make_telemetry_ccsds_packet(apid, data)


def format_telemetry_event(event: TelemetryEventMsg, apid: int) -> "CcsdsPacket":
    """Serialise a TelemetryEventMsg into a CCSDS telemetry packet.

    # TODO: replace JSON encoding with a compact binary TLM format.
    """
    payload: dict[str, object] = {
        "timestamp_utc": event.timestamp_utc,
        "subsystem": event.subsystem,
        "event_name": event.event_name,
        "payload": event.payload,
    }
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return _make_telemetry_ccsds_packet(apid, data)


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
    HEALTH_EMIT_INTERVAL_S: float = 60.0
    last_heartbeat: float = time.monotonic()
    _last_health_emit_time: float = time.monotonic()
    sequence: int = 0

    # Valid keys for the health snapshot accumulator.
    _SNAPSHOT_KEYS: frozenset[str] = frozenset({
        "system_mode", "gimbal_state", "active_faults",
        "frames_captured_today", "bytes_downlinked_today",
        "bytes_remaining_today", "model_version",
        "model_deploy_state", "inference_latency_ms_mean",
        "inference_latency_ms_max", "storage_bytes_used",
    })

    _health_accumulator: dict[str, object] = {
        "system_mode": SystemMode.IDLE,
        "gimbal_state": GimbalState.IDLE,
        "active_faults": [],
        "frames_captured_today": 0,
        "bytes_downlinked_today": 0,
        "bytes_remaining_today": 0,
        "model_version": "unknown",
        "model_deploy_state": ModelDeployState.ACTIVE,
        "inference_latency_ms_mean": 0.0,
        "inference_latency_ms_max": 0.0,
        "storage_bytes_used": 0,
    }

    log.info("telemetry_process_started", apid=apid)

    while True:
        try:
            event: TelemetryEventMsg = telemetry_queue.get(
                timeout=1.0,
            )
        except queue.Empty:
            _maybe_send_heartbeat(
                heartbeat_queue, last_heartbeat,
                HEARTBEAT_INTERVAL_S, sequence,
            )
            # Check if health snapshot is due even on empty queue.
            if (time.monotonic() - _last_health_emit_time
                    >= HEALTH_EMIT_INTERVAL_S):
                _last_health_emit_time = _emit_health_snapshot(
                    _health_accumulator, downlink_queue,
                )
            continue

        now_str = datetime.now(timezone.utc).isoformat(
            timespec="milliseconds",
        )

        # Update health accumulator from event payload.
        for key, val in event.payload.items():
            if key in _SNAPSHOT_KEYS:
                _health_accumulator[key] = val

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
            crc32=zlib.crc32(payload_bytes) & 0xFFFFFFFF,
            item_id=(
                f"telemetry-{event.subsystem}"
                f"-{event.event_name}-{now_str}"
            ),
        )
        try:
            downlink_queue.put_nowait(downlink_item)
        except queue.Full:
            log.warning(
                "downlink_queue_full", subsystem=event.subsystem,
            )

        # Periodic health snapshot emission.
        if (time.monotonic() - _last_health_emit_time
                >= HEALTH_EMIT_INTERVAL_S):
            _last_health_emit_time = _emit_health_snapshot(
                _health_accumulator, downlink_queue,
            )

        now_mono = time.monotonic()
        if now_mono - last_heartbeat >= HEARTBEAT_INTERVAL_S:
            _emit_heartbeat(heartbeat_queue, sequence)
            last_heartbeat = now_mono
            sequence += 1

        log.debug(
            "telemetry_event_processed",
            subsystem=event.subsystem,
            event=event.event_name,
        )


def _emit_health_snapshot(
    accumulator: dict[str, object],
    downlink_queue: "queue.Queue[DownlinkItemMsg]",
) -> float:
    """Build and enqueue a SystemHealthSnapshot. Returns monotonic time."""
    snapshot = SystemHealthSnapshot(
        timestamp_utc=utc_now_iso(),
        system_mode=accumulator["system_mode"],  # type: ignore[arg-type]
        gimbal_state=accumulator["gimbal_state"],  # type: ignore[arg-type]
        active_faults=frozenset(
            accumulator["active_faults"],  # type: ignore[arg-type]
        ),
        frames_captured_today=accumulator[  # type: ignore[arg-type]
            "frames_captured_today"
        ],
        bytes_downlinked_today=accumulator[  # type: ignore[arg-type]
            "bytes_downlinked_today"
        ],
        bytes_remaining_today=accumulator[  # type: ignore[arg-type]
            "bytes_remaining_today"
        ],
        model_version=accumulator["model_version"],  # type: ignore[arg-type]
        model_deploy_state=accumulator[  # type: ignore[arg-type]
            "model_deploy_state"
        ],
        inference_latency_ms_mean=accumulator[  # type: ignore[arg-type]
            "inference_latency_ms_mean"
        ],
        inference_latency_ms_max=accumulator[  # type: ignore[arg-type]
            "inference_latency_ms_max"
        ],
        storage_bytes_used=accumulator[  # type: ignore[arg-type]
            "storage_bytes_used"
        ],
    )
    payload_bytes: bytes = pickle.dumps(snapshot)
    item = DownlinkItemMsg(
        msg_type=MessageType.DOWNLINK_ITEM,
        timestamp_utc=utc_now_iso(),
        priority=DownlinkPriority.HEALTH_TELEMETRY,
        payload_bytes=payload_bytes,
        crc32=zlib.crc32(payload_bytes) & 0xFFFFFFFF,
        item_id=f"health-snapshot-{utc_now_iso()}",
    )
    try:
        downlink_queue.put_nowait(item)
    except queue.Full:
        log.warning("downlink_queue_full_health_snapshot")
    return time.monotonic()


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
