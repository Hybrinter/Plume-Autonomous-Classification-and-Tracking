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
3. Assert that a GimbalCommandMsg appears on the gimbal queue only after TRACKING is entered.
4. Assert state machine transitions via TelemetryEventMsg on the telemetry queue.
5. Shut down the process cleanly.

Note: This test requires subprocess spawning and the full pact package to be importable
from the subprocess. It is skipped until all process.py entry points are complete.
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="integration: requires subprocess setup — implement after run_controller_process() is complete")
def test_controller_pipeline_idle_to_tracking() -> None:
    """Feed persistence-ramping blobs into the controller; verify IDLE→ACQUIRING→TRACKING.

    Setup:
    - run_controller_process() started in a multiprocessing.Process.
    - InferenceResultMsg values injected directly onto inference_queue.
    - GimbalCommandMsg collected from gimbal_queue.
    - TelemetryEventMsg collected from telemetry_queue.

    Assertions:
    - No GimbalCommandMsg on gimbal_queue after frames 1–2 (ACQUIRING, not TRACKING).
    - At least one GimbalCommandMsg on gimbal_queue after frame 3 (TRACKING).
    - TelemetryEventMsg log shows: IDLE → ACQUIRING (frame 1), ACQUIRING → TRACKING (frame 3).
    - Process joins within 5 seconds of stop signal.

    TODO: implement when run_controller_process() entry point is complete.
    """
    # TODO: implement this test
    # import multiprocessing
    # from pact.controller.process import run_controller_process
    # from pact.types.messages import InferenceResultMsg, GimbalCommandMsg, BlobMeta
    # ... spin up process, inject blobs, assert transitions
    pass
