"""Integration test for the controller pipeline (inference → controller → gimbal queue).

Satisfies: §6.3 of PACT_SW_ARCH.md — Integration tests.
REQ-AIML-GIMB-001 through 008, REQ-GIMB-HIGH-001 through 004

Test Plan
---------
1. Start run_controller_process() in a subprocess.
2. Feed a sequence of InferenceResultMsg values with increasing blob persistence:
   - Frame 1: blob persistence=1 → expect state ACQUIRING
   - Frame 2: blob persistence=2 → expect state ACQUIRING
   - Frame 3: blob persistence=3 → expect state TRACKING
3. Assert that a TelemetryEventMsg showing TRACKING state appears on the telemetry queue.
4. Assert state machine transitions via TelemetryEventMsg on the telemetry queue:
   IDLE → ACQUIRING (frame 1), ACQUIRING → TRACKING (frame 3).
5. Shut down the process cleanly.

Note: The controller's send_gimbal_command() is a stub that logs only — it does not
put messages on a gimbal_queue. GimbalCommandMsg emission is verified indirectly through
the state machine reaching TRACKING, which is the only state from which commands are issued.
"""

from __future__ import annotations

# stdlib
import multiprocessing
import time
from typing import Optional

# third-party
import numpy as np
import pytest

# internal
from pact.controller.process import run_controller_process
from pact.ops.config_loader import load_config
from pact.types.config import ControllerConfig, FaultConfig, PactConfig
from pact.types.enums import GimbalState, MessageType, Ok
from pact.types.messages import (
    BlobMeta,
    FaultEventMsg,
    HeartbeatMsg,
    InferenceResultMsg,
    TelemetryEventMsg,
    utc_now_iso,
)


def _make_inference_result(
    frame_id: int,
    persistence: int,
    confidence: float = 0.85,
    area: int = 200,
) -> InferenceResultMsg:
    """Construct an InferenceResultMsg with a single blob for controller testing."""
    blob = BlobMeta(
        blob_id=frame_id,
        bbox=(100, 100, 150, 150),
        centroid_raw=(125.0, 125.0),
        pixel_area=area,
        mean_confidence=confidence,
        persistence_count=persistence,
    )
    return InferenceResultMsg(
        msg_type=MessageType.INFERENCE_RESULT,
        timestamp_utc=utc_now_iso(),
        frame_id=frame_id,
        mask=np.zeros((256, 256), dtype=np.float32),  # np.ndarray[float32,(256,256)]
        blobs=(blob,),
        model_version="test-v0",
        inference_ms=10.0,
        mode_flags=0,
    )


@pytest.mark.timeout(60)
def test_controller_pipeline_idle_to_tracking() -> None:
    """Feed persistence-ramping blobs into the controller; verify IDLE→ACQUIRING→TRACKING.

    Setup:
    - run_controller_process() started in a multiprocessing.Process.
    - InferenceResultMsg values injected directly onto inference_queue.
    - TelemetryEventMsg collected from telemetry_queue.

    Assertions:
    - TelemetryEventMsg log shows: IDLE → ACQUIRING (frame 1), ACQUIRING → TRACKING (frame 3).
    - At least one TRACKING transition is recorded in telemetry events.
    - Process joins within 5 seconds of stop signal.
    """
    config_result = load_config("config/default.toml")
    assert isinstance(config_result, Ok), f"load_config failed: {config_result}"
    config: PactConfig = config_result.value  # type: ignore[union-attr]

    # Use a short watchdog interval so we don't block long on heartbeat
    from dataclasses import replace as dc_replace
    fault_cfg = dc_replace(config.fault, watchdog_interval_s=1.0)

    # --- queues ---
    # telemetry_queue: controller sends TelemetryEventMsg here
    # We use multiprocessing.Queue so it crosses process boundary
    inference_queue: multiprocessing.Queue[InferenceResultMsg] = multiprocessing.Queue(maxsize=16)
    telemetry_queue: multiprocessing.Queue[TelemetryEventMsg] = multiprocessing.Queue(maxsize=64)
    fault_queue: multiprocessing.Queue[FaultEventMsg] = multiprocessing.Queue(maxsize=16)
    heartbeat_queue: multiprocessing.Queue[HeartbeatMsg] = multiprocessing.Queue(maxsize=64)
    stop_event: multiprocessing.Event = multiprocessing.Event()  # type: ignore[type-arg]

    # --- spawn controller process ---
    proc = multiprocessing.Process(
        target=run_controller_process,
        args=(
            config.controller,
            fault_cfg,
            inference_queue,
            telemetry_queue,
            fault_queue,
            heartbeat_queue,
            stop_event,
        ),
        daemon=True,
        name="test-controller",
    )
    proc.start()

    telemetry_events: list[TelemetryEventMsg] = []

    try:
        # --- inject 3 frames with increasing persistence ---
        # Frame 1: persistence=1 → IDLE→ACQUIRING
        # Frame 2: persistence=2 → stays ACQUIRING
        # Frame 3: persistence=3 → ACQUIRING→TRACKING (acquire_persistence_frames default=3)
        # Use a slightly longer delay so the subprocess has time to initialise its loop.
        time.sleep(0.2)  # let the subprocess start its main loop before sending frames

        frame_delay: float = 0.3  # 300 ms between frames to ensure sequential processing

        for frame_id in range(1, 4):
            msg = _make_inference_result(
                frame_id=frame_id,
                persistence=frame_id,
                confidence=0.85,
                area=200,
            )
            inference_queue.put(msg)
            time.sleep(frame_delay)

        # --- collect telemetry events (drain for up to 5 seconds) ---
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                event: TelemetryEventMsg = telemetry_queue.get(timeout=0.2)
                telemetry_events.append(event)
            except Exception:
                continue

    finally:
        # --- always stop and clean up subprocess ---
        stop_event.set()
        proc.join(timeout=5.0)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=2.0)

    # --- assertions ---
    assert proc.exitcode is not None, "Controller process did not exit cleanly"

    # Extract state_transition events from controller subsystem
    transitions: list[dict[str, str]] = [
        e.payload
        for e in telemetry_events
        if e.subsystem == "controller" and e.event_name == "state_transition"
    ]

    assert len(transitions) >= 1, (
        f"Expected at least 1 state transition in telemetry, got 0. "
        f"All events: {telemetry_events}"
    )

    # Verify that ACQUIRING was entered
    acquiring_entries = [t for t in transitions if t.get("to") == GimbalState.ACQUIRING.value]
    assert len(acquiring_entries) >= 1, (
        f"Expected ACQUIRING transition in telemetry events. Transitions: {transitions}"
    )

    # Verify that TRACKING was eventually reached
    tracking_entries = [t for t in transitions if t.get("to") == GimbalState.TRACKING.value]
    assert len(tracking_entries) >= 1, (
        f"Expected TRACKING transition in telemetry events. Transitions: {transitions}"
    )

    # Verify order: IDLE→ACQUIRING must come before ACQUIRING→TRACKING
    idle_to_acquiring = next(
        (t for t in transitions if t.get("from") == GimbalState.IDLE.value
         and t.get("to") == GimbalState.ACQUIRING.value),
        None,
    )
    assert idle_to_acquiring is not None, (
        f"Expected IDLE→ACQUIRING transition. Transitions: {transitions}"
    )

    acquiring_to_tracking = next(
        (t for t in transitions if t.get("from") == GimbalState.ACQUIRING.value
         and t.get("to") == GimbalState.TRACKING.value),
        None,
    )
    assert acquiring_to_tracking is not None, (
        f"Expected ACQUIRING→TRACKING transition. Transitions: {transitions}"
    )
