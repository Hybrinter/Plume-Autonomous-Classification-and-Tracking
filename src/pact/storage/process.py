"""Storage process entry point.

Runs as a threading.Thread inside the storage OS process.  Drains StorageWriteMsg from
the storage_queue, calls write_frame() + append_to_manifest(), then enqueues a
DownlinkItemMsg (SCIENCE_DATA priority) on the downlink_queue.

Satisfies: REQ-IMAG-HIGH-003, GOAL-003, GOAL-004.
"""

from __future__ import annotations

# stdlib
import multiprocessing
import queue
import threading
import time
from datetime import datetime, timezone
from typing import Optional

# internal
from pact.types.config import StorageConfig
from pact.types.enums import DownlinkPriority, FaultCode, MessageType
from pact.types.enums import Ok, Err  # type: ignore[attr-defined]
from pact.types.messages import (
    DownlinkItemMsg,
    FaultEventMsg,
    HeartbeatMsg,
    StorageWriteMsg,
)
from pact.storage.manifest import append_to_manifest
from pact.storage.writer import write_frame

import structlog

log = structlog.get_logger().bind(subsystem="storage")


def run_storage_process(
    config: StorageConfig,
    storage_queue: "multiprocessing.Queue[StorageWriteMsg]",
    downlink_queue: "queue.Queue[DownlinkItemMsg]",
    fault_queue: "multiprocessing.Queue[FaultEventMsg]",
    heartbeat_queue: "multiprocessing.Queue[HeartbeatMsg]",
    manifest_path: str,
    stop_event: Optional[threading.Event] = None,
) -> None:
    """Storage process main loop.

    Drains StorageWriteMsg items from storage_queue, writes each frame to disk,
    appends to the manifest, and enqueues a DownlinkItemMsg for each stored frame.
    Emits a HeartbeatMsg every config.watchdog_interval_s seconds (approximated by
    the loop timeout).

    This function is intended to be the target of a threading.Thread.  The caller
    is responsible for providing a threading.Event for clean shutdown.

    # TODO: implement actual heartbeat timing via a separate timer thread
    # TODO: implement downlink payload serialisation (currently empty bytes placeholder)
    """
    # TODO: set real heartbeat interval from FaultConfig (not available here — pass it in)
    HEARTBEAT_INTERVAL_S: float = 5.0
    last_heartbeat: float = time.monotonic()
    sequence: int = 0
    _stop = stop_event if stop_event is not None else threading.Event()

    log.info("storage_process_started", manifest_path=manifest_path)

    while not _stop.is_set():
        # --- drain queue with timeout ---
        try:
            msg: StorageWriteMsg = storage_queue.get(timeout=1.0)
        except Exception:
            # queue.Empty or multiprocessing.Queue timeout
            _maybe_heartbeat(heartbeat_queue, last_heartbeat, HEARTBEAT_INTERVAL_S, sequence)
            continue

        now_str = datetime.now(timezone.utc).isoformat(timespec="milliseconds")

        # --- write frame ---
        result = write_frame(msg, config.data_root)
        if isinstance(result, Err):
            log.error(
                "write_frame_failed",
                frame_id=msg.frame_id,
                fault_code=result.error.value,
            )
            fault_queue.put(
                FaultEventMsg(
                    msg_type=MessageType.FAULT_EVENT,
                    timestamp_utc=now_str,
                    fault_code=result.error,
                    subsystem="storage",
                    detail=f"write_frame failed for frame_id={msg.frame_id}",
                )
            )
            continue

        record = result.value

        # --- append to manifest ---
        manifest_result = append_to_manifest(record, manifest_path)
        if isinstance(manifest_result, Err):
            log.error(
                "manifest_append_failed",
                frame_id=msg.frame_id,
                fault_code=manifest_result.error.value,
            )
            fault_queue.put(
                FaultEventMsg(
                    msg_type=MessageType.FAULT_EVENT,
                    timestamp_utc=now_str,
                    fault_code=manifest_result.error,
                    subsystem="storage",
                    detail=f"manifest append failed for frame_id={msg.frame_id}",
                )
            )
            continue

        log.info("frame_stored", frame_id=msg.frame_id, usability=record.usability.value)

        # --- enqueue downlink item ---
        # TODO: serialise record into a proper CCSDS science data payload
        downlink_item = DownlinkItemMsg(
            msg_type=MessageType.DOWNLINK_ITEM,
            timestamp_utc=now_str,
            priority=DownlinkPriority.SCIENCE_DATA,
            payload_bytes=b"",          # placeholder — see TODO above
            crc32=0,                    # placeholder
            item_id=f"frame-{msg.frame_id:08d}",
        )
        try:
            downlink_queue.put_nowait(downlink_item)
        except queue.Full:
            log.warning("downlink_queue_full", frame_id=msg.frame_id)

        # --- heartbeat ---
        now_mono = time.monotonic()
        if now_mono - last_heartbeat >= HEARTBEAT_INTERVAL_S:
            _emit_heartbeat(heartbeat_queue, sequence)
            last_heartbeat = now_mono
            sequence += 1

    log.info("storage_process_stopped")


def _emit_heartbeat(
    heartbeat_queue: "multiprocessing.Queue[HeartbeatMsg]",
    sequence: int,
) -> None:
    now_str = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    try:
        heartbeat_queue.put_nowait(
            HeartbeatMsg(
                msg_type=MessageType.HEARTBEAT,
                timestamp_utc=now_str,
                subsystem="storage",
                sequence=sequence,
            )
        )
    except Exception:
        log.warning("heartbeat_queue_full")


def _maybe_heartbeat(
    heartbeat_queue: "multiprocessing.Queue[HeartbeatMsg]",
    last_heartbeat: float,
    interval_s: float,
    sequence: int,
) -> float:
    """Emit a heartbeat if enough time has elapsed. Returns updated last_heartbeat time."""
    now_mono = time.monotonic()
    if now_mono - last_heartbeat >= interval_s:
        _emit_heartbeat(heartbeat_queue, sequence)
        return now_mono
    return last_heartbeat
