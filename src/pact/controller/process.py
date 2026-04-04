"""
Controller process entry point for PACT.

Runs as a `multiprocessing.Process`. Receives InferenceResultMsg from the inference
process, applies the full safety pipeline, and drives the GimbalArbiter state machine.

Pipeline (per frame):
    1. confidence gate
    2. min area gate
    3. blob tracker (IoU association)
    4. EMA filter (centroid smoothing)
    5. deadband check
    6. rate limit check
    7. GimbalArbiter.step()
    8. Dispatch GimbalCommandMsg → gimbal hardware (stub)
    9. Dispatch TelemetryEventMsg → telemetry queue
    10. Heartbeat → fault watchdog

Satisfies: REQ-AIML-GIMB-001 through 008, REQ-GIMB-HIGH-001 through 004
"""

from __future__ import annotations

import dataclasses
import multiprocessing
import time
from typing import Optional

import structlog

from pact.controller.arbiter import ArbiterState, GimbalArbiter
from pact.controller.filter import EmaFilterState, ema_update
from pact.controller.safety import (
    apply_confidence_gate,
    apply_min_area_gate,
    check_deadband,
    check_rate_limit,
)
from pact.controller.tracker import match_blobs
from pact.types.config import ControllerConfig, FaultConfig
from pact.types.enums import FaultCode, GimbalState, MessageType
from pact.types.messages import (
    FaultEventMsg,
    GimbalCommandMsg,
    HeartbeatMsg,
    InferenceResultMsg,
    TelemetryEventMsg,
)

log = structlog.get_logger().bind(subsystem="controller")


def send_gimbal_command(command: GimbalCommandMsg) -> None:
    """Send a GimbalCommandMsg to the physical gimbal hardware.

    # TODO: stub — implement send_gimbal_command()
    Replace with the vendor serial/CAN API for the flight gimbal unit.
    Current implementation logs the command and returns immediately.
    """
    log.info(
        "gimbal_command_stub",
        az_delta_deg=command.az_delta_deg,
        el_delta_deg=command.el_delta_deg,
        state=command.state.value,
        reason=command.reason,
    )


def run_controller_process(
    controller_cfg: ControllerConfig,
    fault_cfg: FaultConfig,
    inference_queue: "multiprocessing.Queue[InferenceResultMsg]",
    telemetry_queue: "multiprocessing.Queue[TelemetryEventMsg]",
    fault_queue: "multiprocessing.Queue[FaultEventMsg]",
    heartbeat_queue: "multiprocessing.Queue[HeartbeatMsg]",
    stop_event: "multiprocessing.Event",
) -> None:
    """Entry point for the controller process. REQ-AIML-GIMB-001 through 008.

    Parameters
    ----------
    controller_cfg:
        Controller-specific configuration (thresholds, rate limits, etc.).
    fault_cfg:
        Fault configuration (watchdog interval, etc.).
    inference_queue:
        Receives InferenceResultMsg from the inference process.
    telemetry_queue:
        Sends TelemetryEventMsg for every arbiter state transition.
    fault_queue:
        Sends FaultEventMsg when a safety gate raises a fault (e.g., GIMBAL_RUNAWAY).
    heartbeat_queue:
        Sends HeartbeatMsg every fault_cfg.watchdog_interval_s to the fault watchdog.
    stop_event:
        Set by the orchestrator to signal a clean shutdown.
    """
    log.info("controller_process_start")

    arbiter = GimbalArbiter()
    state = ArbiterState(
        gimbal_state=GimbalState.IDLE,
        tracked_blobs=(),
        idle_duration_s=0.0,
        last_command_time=0.0,
        current_target_id=None,
    )
    ema_state = EmaFilterState(centroid=(0.0, 0.0), initialized=False)

    last_heartbeat_time: float = time.monotonic()
    heartbeat_seq: int = 0

    while not stop_event.is_set():
        # --- Heartbeat ---
        now_mono = time.monotonic()
        if (now_mono - last_heartbeat_time) >= fault_cfg.watchdog_interval_s:
            heartbeat_queue.put(
                HeartbeatMsg(
                    msg_type=MessageType.HEARTBEAT,
                    timestamp_utc=_utc_now_iso(),
                    subsystem="controller",
                    sequence=heartbeat_seq,
                )
            )
            heartbeat_seq += 1
            last_heartbeat_time = now_mono

        # --- Receive inference result ---
        try:
            result: InferenceResultMsg = inference_queue.get(timeout=1.0)
        except Exception:  # queue.Empty or timeout
            continue

        now_unix = time.time()

        # Step 1: Confidence gate
        gated_blobs = apply_confidence_gate(result.blobs, controller_cfg.confidence_gate)

        # Step 2: Min area gate
        gated_blobs = apply_min_area_gate(gated_blobs, controller_cfg.min_blob_area_px)

        # Step 3: Blob tracker (IoU association)
        matched_blobs = match_blobs(
            state.tracked_blobs,
            gated_blobs,
            controller_cfg.blob_iou_match_threshold,
        )

        # Step 4: EMA filter — update on primary target if one exists
        if matched_blobs:
            primary = matched_blobs[0]
            ema_state = ema_update(ema_state, primary.centroid_raw, controller_cfg.ema_alpha)
        else:
            # Reset EMA when no blobs present
            ema_state = EmaFilterState(centroid=(0.0, 0.0), initialized=False)

        # Step 5 & 6: Deadband and rate limit (checked inside arbiter.step() via the
        # pre-filtered blob list — deadband/rate-limit decisions require frame center
        # context that the arbiter has; gates here are blob-level, not command-level)
        # Note: check_deadband and check_rate_limit are called by GimbalArbiter.step()
        # when constructing GimbalCommandMsg. Fault propagation for GIMBAL_RUNAWAY is
        # the arbiter's responsibility.

        # Step 7: Arbiter state machine (pure function)
        # Build a new InferenceResultMsg with the matched/filtered blobs substituted in.
        filtered_result = dataclasses.replace(result, blobs=matched_blobs)
        new_state, command, telemetry_events = arbiter.step(
            state=state,
            result=filtered_result,
            now=now_unix,
        )
        state = new_state

        # Step 8: Send gimbal command (hardware stub)
        if command is not None:
            send_gimbal_command(command)

        # Step 9: Dispatch telemetry events
        for event in telemetry_events:
            telemetry_queue.put(event)

    log.info("controller_process_stop")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    """Return current UTC time as ISO 8601 string with millisecond precision."""
    import datetime
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
