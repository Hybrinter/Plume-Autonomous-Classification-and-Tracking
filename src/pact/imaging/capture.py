"""
Frame capture loop for PACT imaging subsystem.

Provides the tight acquisition loop that continuously reads frames from an AbstractCamera
and places them on the raw frame queue for downstream processing. Handles stall detection
and emits FaultEventMsg when a stall is detected.

Satisfies: REQ-AIML-IMAG-001, REQ-AIML-IMAG-002
"""

from __future__ import annotations

import queue
import time

import structlog

from pact.imaging.camera import AbstractCamera
from pact.types.config import FaultConfig
from pact.types.enums import FaultCode, MessageType, Ok
from pact.types.messages import FaultEventMsg, RawFrameMsg, utc_now_iso

log = structlog.get_logger().bind(subsystem="imaging")


def run_capture_loop(
    camera: AbstractCamera,
    out_queue: "queue.Queue[RawFrameMsg]",
    fault_queue: "queue.Queue[FaultEventMsg]",
    fault_cfg: FaultConfig,
    stop_event: "object",  # threading.Event — typed as object to avoid circular import issues
    stall_timeout_s: float = 5.0,
) -> None:
    """Tight frame acquisition loop. Stub entry point.

    Acquires frames from `camera` as fast as the hardware allows and places each
    RawFrameMsg on `out_queue`. If no frame is received within `stall_timeout_s`,
    a FaultEventMsg with FaultCode.CAMERA_STALL is placed on `fault_queue`.

    The loop runs until `stop_event.is_set()` returns True.

    Parameters
    ----------
    camera:
        Any AbstractCamera implementation. In production: FlirBlackflyCamera.
        In tests: MockCamera.
    out_queue:
        Thread-safe queue for delivering RawFrameMsg to the preprocessing/inference chain.
    fault_queue:
        Thread-safe queue for delivering FaultEventMsg to the fault process.
    fault_cfg:
        Fault configuration (unused currently; stall_timeout_s is the operative parameter).
    stop_event:
        threading.Event. Set by the orchestrator to signal a clean shutdown.
    stall_timeout_s:
        Maximum seconds between frames before a stall fault is raised.
        Should come from config in production; provided as a parameter here for testability.
    """
    log.info("capture_loop_start", stall_timeout_s=stall_timeout_s)
    last_frame_time = time.monotonic()

    while not stop_event.is_set():  # type: ignore[union-attr]
        result = camera.acquire_frame()

        if isinstance(result, Ok):
            frame: RawFrameMsg = result.value
            last_frame_time = time.monotonic()
            try:
                out_queue.put_nowait(frame)
            except queue.Full:
                log.warning("capture_queue_full", frame_id=frame.frame_id)
        else:  # Err — camera returned a fault (e.g., end-of-stream from MockCamera)
            elapsed = time.monotonic() - last_frame_time
            if elapsed >= stall_timeout_s:
                log.error("camera_stall_detected", elapsed_s=elapsed)
                fault_queue.put(
                    FaultEventMsg(
                        msg_type=MessageType.FAULT_EVENT,
                        timestamp_utc=utc_now_iso(),
                        fault_code=FaultCode.CAMERA_STALL,
                        subsystem="imaging",
                        detail=f"No frame received for {elapsed:.1f}s (limit {stall_timeout_s}s)",
                    )
                )
                last_frame_time = time.monotonic()  # reset to avoid repeated fault floods

    log.info("capture_loop_stop")


