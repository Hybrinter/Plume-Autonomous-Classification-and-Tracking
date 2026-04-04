"""End-to-end smoke test for the full PACT pipeline.

Satisfies: §6.4 of PACT_SW_ARCH.md — End-to-end pipeline smoke test.
Must complete in under 60 seconds (enforced by pytest-timeout).
Mark: @pytest.mark.e2e

Test Plan
---------
This test exercises the entire system as it would run on the Jetson Xavier:

Setup:
  1. Load config/default.toml.
  2. Instantiate MockCamera configured to emit 10 synthetic frames, 3 of which carry a
     synthetic plume blob above the confidence threshold.
  3. Instantiate a randomly initialized PactSegmentationModel (no real weights required).
  4. Wire all subsystems together using the same queue topology as ops/main.py.
  5. Inject synthetic InferenceResultMsg values for frames 1–3 directly onto the
     inference_queue, bypassing the real model output (see tests/CLAUDE.md §5).

Assertions (all 10 must pass):
  1. All 10 frames are consumed from the imaging queue within the timeout.
  2. All 10 frames produce an InferenceResultMsg on the inference result queue.
  3. Frames 1–2 produce GimbalState.ACQUIRING (blob present, persistence < 3).
  4. Frame 3+ produces at least one GimbalState.TRACKING entry in arbiter state log.
  5. At least one GimbalCommandMsg is emitted while in TRACKING state.
  6. All 10 frames produce a StorageWriteMsg that lands on the storage queue.
  7. At least 10 TelemetryEventMsg entries appear on the telemetry queue.
  8. No FaultEventMsg with a non-NONE fault code is emitted during normal operation.
  9. The heartbeat watchdog receives at least one HeartbeatMsg from each active subsystem.
 10. After all frames are processed, the system shuts down cleanly (all processes join
     within 5 seconds).

Injection Point Note:
  The randomly initialized model produces garbage segmentation — blobs from real inference
  output will not meet the confidence threshold. Assertions 3–5 depend on synthetic
  InferenceResultMsg values injected directly onto inference_queue for frames 1–3.
  See tests/CLAUDE.md §5 for the full explanation of this injection point and why it is
  the correct place to decouple model output quality from controller correctness testing.
"""

from __future__ import annotations

import pytest


@pytest.mark.e2e
def test_full_pipeline_smoke() -> None:
    """End-to-end smoke test: 10 synthetic frames through the full PACT pipeline.

    This test is currently skipped. Implement after all process.py entry points are
    complete and the queue topology in ops/main.py is wired.

    See module docstring for the complete test plan and all 10 assertions.
    """
    pytest.skip(
        "e2e: implement after all process.py entry points complete "
        "(imaging, inference, controller, storage, comms, fault, telemetry)"
    )

    # --- Setup (implement below when unblocked) ---
    # TODO: 1. Load config/default.toml via load_config()
    # TODO: 2. Create MockCamera with 10 frames (3 containing synthetic blobs)
    # TODO: 3. Build randomly initialized PactSegmentationModel
    # TODO: 4. Create all queues (raw_frame, inference, gimbal, telemetry, storage,
    #          fault, heartbeat, downlink, uplink, mode)
    # TODO: 5. Spawn all processes using the ops/main.py topology
    # TODO: 6. Set up InferenceResultMsg injection shim for frames 1–3

    # --- Assertions (implement below when unblocked) ---
    # TODO: Assertion 1 — all 10 frames consumed from imaging queue within timeout
    # TODO: Assertion 2 — all 10 frames produce InferenceResultMsg
    # TODO: Assertion 3 — frames 1–2: arbiter state log shows GimbalState.ACQUIRING
    # TODO: Assertion 4 — frame 3+: arbiter state log shows GimbalState.TRACKING
    # TODO: Assertion 5 — at least one GimbalCommandMsg emitted from TRACKING state
    # TODO: Assertion 6 — all 10 frames produce StorageWriteMsg on storage queue
    # TODO: Assertion 7 — at least 10 TelemetryEventMsg on telemetry queue
    # TODO: Assertion 8 — no FaultEventMsg with non-NONE fault code during normal op
    # TODO: Assertion 9 — heartbeat watchdog receives HeartbeatMsg from each subsystem
    # TODO: Assertion 10 — clean shutdown: all processes join within 5 seconds
