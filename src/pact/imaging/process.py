"""
Imaging process entry point for PACT.

Spawns the camera capture loop inside a daemon thread and routes frames to the
preprocessing/inference pipeline queue. Runs as a `threading.Thread` (not a process)
because frame capture is I/O-bound over GigE Vision.

See imaging/CLAUDE.md for concurrency rationale.

Satisfies: REQ-AIML-IMAG-001, REQ-AIML-IMAG-002
"""

from __future__ import annotations

import queue
import threading

import structlog

from pact.imaging.camera import AbstractCamera, FlirBlackflyCamera
from pact.imaging.capture import run_capture_loop
from pact.types.config import FaultConfig
from pact.types.enums import MessageType
from pact.types.messages import (
    FaultEventMsg,
    HeartbeatMsg,
    RawFrameMsg,
    utc_now_iso,
)

log = structlog.get_logger().bind(subsystem="imaging")

_heartbeat_sequence: int = 0


def _heartbeat_loop(
    heartbeat_queue: "queue.Queue[HeartbeatMsg]",
    fault_cfg: FaultConfig,
    stop_event: threading.Event,
) -> None:
    """Send HeartbeatMsg every watchdog_interval_s until stop_event is set."""
    global _heartbeat_sequence
    while not stop_event.is_set():
        msg = HeartbeatMsg(
            msg_type=MessageType.HEARTBEAT,
            timestamp_utc=utc_now_iso(),
            subsystem="imaging",
            sequence=_heartbeat_sequence,
        )
        _heartbeat_sequence += 1
        try:
            heartbeat_queue.put_nowait(msg)
        except queue.Full:
            pass
        stop_event.wait(timeout=fault_cfg.watchdog_interval_s)


def run_imaging_process(
    fault_cfg: FaultConfig,
    raw_frame_queue: "queue.Queue[RawFrameMsg]",
    fault_queue: "queue.Queue[FaultEventMsg]",
    heartbeat_queue: "queue.Queue[HeartbeatMsg]",
    stop_event: threading.Event,
    camera: "AbstractCamera | None" = None,
) -> None:
    """Entry point for the imaging subsystem. Stub.

    Initialises the camera (FlirBlackflyCamera in production, or `camera` if provided
    for testing/injection), starts the capture loop in a daemon thread, and monitors
    for shutdown.

    Parameters
    ----------
    fault_cfg:
        Fault configuration (watchdog interval, stall timeout).
    raw_frame_queue:
        Delivers RawFrameMsg to the inference process.
    fault_queue:
        Receives FaultEventMsg from the capture loop (stall detection).
    heartbeat_queue:
        Sends HeartbeatMsg to the fault watchdog.
    stop_event:
        threading.Event. Set by the orchestrator to request a clean shutdown.
    camera:
        Optional AbstractCamera override. If None, FlirBlackflyCamera is instantiated.
        Provide a MockCamera here in tests and integration scenarios.
    """
    log.info("imaging_process_start")

    # Resolve camera — use injected instance if provided (test injection point)
    active_camera: AbstractCamera
    if camera is not None:
        active_camera = camera
    else:
        # TODO: stub — pass serial_number from ImagingConfig when config dataclass is added
        active_camera = FlirBlackflyCamera()

    start_result = active_camera.start_acquisition()
    if hasattr(start_result, "error"):
        log.error("camera_start_failed", error=start_result.error)  # type: ignore[union-attr]
        return

    capture_thread = threading.Thread(
        target=run_capture_loop,
        args=(active_camera, raw_frame_queue, fault_queue, fault_cfg, stop_event),
        daemon=True,
        name="imaging-capture",
    )
    capture_thread.start()
    log.info("capture_thread_started")

    heartbeat_thread = threading.Thread(
        target=_heartbeat_loop,
        args=(heartbeat_queue, fault_cfg, stop_event),
        daemon=True,
        name="imaging-heartbeat",
    )
    heartbeat_thread.start()
    log.info("heartbeat_thread_started")

    capture_thread.join()
    stop_event.set()
    heartbeat_thread.join(timeout=2.0)
    active_camera.stop_acquisition()
    log.info("imaging_process_stop")
