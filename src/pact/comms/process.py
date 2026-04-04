"""
Comms process entry point for PACT.

Runs as an asyncio-based subsystem (not a multiprocessing.Process — see comms/CLAUDE.md
for rationale). In the current stub implementation it is invoked from a dedicated thread
or process by ops/main.py. The asyncio event loop is started inside this function.

Responsibilities:
- Drain the downlink priority queue during open comm windows, respecting the daily budget.
- Receive uplink chunks and route to the model upload session handler.
- Monitor for comm timeout faults.
- Send heartbeat to the fault watchdog.

Satisfies: REQ-COMM-HIGH-001, REQ-COMM-HIGH-002, REQ-COMM-HIGH-003, GOAL-004, GOAL-008
"""

from __future__ import annotations

import asyncio
import datetime
import queue
from typing import Optional

import structlog

from pact.comms.downlink import DownlinkQueue
from pact.comms.uplink import ModelUploadSession, process_uplink_chunk
from pact.types.config import CommsConfig, FaultConfig
from pact.types.enums import FaultCode, MessageType, ModelDeployState
from pact.types.messages import (
    DownlinkItemMsg,
    FaultEventMsg,
    HeartbeatMsg,
    UploadChunkMsg,
)

log = structlog.get_logger().bind(subsystem="comms")


def run_comms_process(
    comms_cfg: CommsConfig,
    fault_cfg: FaultConfig,
    downlink_in_queue: "queue.Queue[DownlinkItemMsg]",
    uplink_queue: "queue.Queue[UploadChunkMsg]",
    fault_queue: "queue.Queue[FaultEventMsg]",
    heartbeat_queue: "queue.Queue[HeartbeatMsg]",
    stop_event: "object",  # threading.Event or multiprocessing.Event
) -> None:
    """Entry point for the comms subsystem. Stub.

    Starts an asyncio event loop and runs the comms coroutine until stop_event is set.

    Parameters
    ----------
    comms_cfg:
        Comms-specific configuration (rate limits, byte budgets, APID, etc.).
    fault_cfg:
        Fault configuration (watchdog interval).
    downlink_in_queue:
        Receives DownlinkItemMsg from storage and telemetry subsystems.
    uplink_queue:
        Receives UploadChunkMsg from the ground station (via radio interface stub).
    fault_queue:
        Sends FaultEventMsg on comm timeout or budget overrun detection.
    heartbeat_queue:
        Sends HeartbeatMsg to the fault watchdog.
    stop_event:
        threading.Event (or compatible). Set by the orchestrator to request shutdown.
    """
    log.info("comms_process_start")
    asyncio.run(_comms_main(
        comms_cfg=comms_cfg,
        fault_cfg=fault_cfg,
        downlink_in_queue=downlink_in_queue,
        uplink_queue=uplink_queue,
        fault_queue=fault_queue,
        heartbeat_queue=heartbeat_queue,
        stop_event=stop_event,
    ))
    log.info("comms_process_stop")


async def _comms_main(
    comms_cfg: CommsConfig,
    fault_cfg: FaultConfig,
    downlink_in_queue: "queue.Queue[DownlinkItemMsg]",
    uplink_queue: "queue.Queue[UploadChunkMsg]",
    fault_queue: "queue.Queue[FaultEventMsg]",
    heartbeat_queue: "queue.Queue[HeartbeatMsg]",
    stop_event: "object",
) -> None:
    """Asyncio main coroutine for the comms subsystem. Stub.

    Runs a polling loop that:
    1. Drains downlink_in_queue into the priority DownlinkQueue.
    2. Dequeues and transmits items if the comm window is open and budget permits.
    3. Processes uplink chunks from uplink_queue.
    4. Sends heartbeats every fault_cfg.watchdog_interval_s.

    # TODO: stub — replace polling with asyncio.Queue and proper event-driven approach
    # once the radio interface provides an async socket/stream.
    """
    dl_queue = DownlinkQueue(
        daily_limit_bytes=comms_cfg.max_daily_downlink_bytes,
        allowed_comm_days=comms_cfg.comm_window_days,
    )

    upload_session: Optional[ModelUploadSession] = None
    heartbeat_seq: int = 0
    last_heartbeat: float = asyncio.get_event_loop().time()

    while not stop_event.is_set():  # type: ignore[union-attr]
        now = asyncio.get_event_loop().time()

        # --- Heartbeat ---
        if (now - last_heartbeat) >= fault_cfg.watchdog_interval_s:
            heartbeat_queue.put_nowait(
                HeartbeatMsg(
                    msg_type=MessageType.HEARTBEAT,
                    timestamp_utc=_utc_now_iso(),
                    subsystem="comms",
                    sequence=heartbeat_seq,
                )
            )
            heartbeat_seq += 1
            last_heartbeat = now

        # --- Drain incoming downlink items into priority queue ---
        try:
            while True:
                item: DownlinkItemMsg = downlink_in_queue.get_nowait()
                dl_queue.enqueue(item)
        except queue.Empty:
            pass

        # --- Dequeue and transmit ---
        item_to_send = dl_queue.dequeue(utc_now=datetime.datetime.utcnow())
        if item_to_send is not None:
            await _transmit_downlink_item(item_to_send)

        # --- Process uplink chunks ---
        try:
            chunk: UploadChunkMsg = uplink_queue.get_nowait()
            if upload_session is not None:
                result = process_uplink_chunk(upload_session, chunk)
                if hasattr(result, "value"):
                    upload_session = result.value  # type: ignore[union-attr]
                else:
                    log.error("uplink_chunk_failed", error=result.error)  # type: ignore[union-attr]
                    fault_queue.put_nowait(
                        FaultEventMsg(
                            msg_type=MessageType.FAULT_EVENT,
                            timestamp_utc=_utc_now_iso(),
                            fault_code=FaultCode.MODEL_CORRUPT,
                            subsystem="comms",
                            detail="Uplink chunk CRC verification failed",
                        )
                    )
        except queue.Empty:
            pass

        # Yield control to the event loop
        await asyncio.sleep(0.05)


async def _transmit_downlink_item(item: DownlinkItemMsg) -> None:
    """Transmit a DownlinkItemMsg to the TDRSS radio interface.

    # TODO: stub — write to file/socket placeholder. Replace with vendor TDRSS modem API.
    """
    log.info(
        "downlink_transmit_stub",
        item_id=item.item_id,
        priority=item.priority.value,
        payload_bytes=len(item.payload_bytes),
    )


def _utc_now_iso() -> str:
    """Return current UTC time as ISO 8601 string with millisecond precision."""
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
