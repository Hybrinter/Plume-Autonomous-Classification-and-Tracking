"""Demo: simulate 20 frames through the GimbalArbiter state machine.

Simulates 20 synthetic InferenceResultMsg values with a ramp-up in blob persistence
followed by a loss of target, then feeds them through GimbalArbiter.step() frame by frame.
Prints the arbiter state at each frame.

Expected output:
  Frame  1: IDLE → ACQUIRING  (first blob detected)
  Frame  2: ACQUIRING         (persistence=2, not yet ≥ 3)
  Frame  3: ACQUIRING → TRACKING (persistence=3)
  Frames 4–10: TRACKING       (blob persists above threshold)
  Frame 11: TRACKING → IDLE   (blob lost)
  Frames 12–20: IDLE          (no blob)

Usage
-----
    python scripts/demo_controller.py

Satisfies: §7 of PACT_SW_ARCH.md (scripts/demo_controller.py)
This script is fully functional since GimbalArbiter is pure logic with no I/O.
"""

from __future__ import annotations

# stdlib
import time
from typing import Optional

# third-party
import numpy as np

# internal
from pact.controller.arbiter import ArbiterState, GimbalArbiter
from pact.types.enums import GimbalState, MessageType
from pact.types.messages import BlobMeta, InferenceResultMsg


def make_blob(persistence_count: int, blob_id: int = 1) -> BlobMeta:
    """Create a synthetic blob above all safety gates."""
    return BlobMeta(
        blob_id=blob_id,
        bbox=(100, 100, 150, 150),
        centroid_raw=(125.0, 125.0),
        pixel_area=200,
        mean_confidence=0.85,
        persistence_count=persistence_count,
    )


def make_result(blobs: tuple[BlobMeta, ...], frame_id: int) -> InferenceResultMsg:
    """Create a synthetic InferenceResultMsg (blobs pre-filtered through all gates)."""
    return InferenceResultMsg(
        msg_type=MessageType.INFERENCE_RESULT,
        timestamp_utc="2026-04-03T00:00:00.000Z",
        frame_id=frame_id,
        mask=np.zeros((256, 256), dtype=np.float32),
        blobs=blobs,
        model_version="demo-v0",
        inference_ms=50.0,
        mode_flags=0,
    )


def main() -> None:
    """Simulate 20 frames and print arbiter state at each step."""
    print("PACT GimbalArbiter Demo")
    print("=" * 60)
    print(f"{'Frame':>5}  {'State':15}  {'Blobs':>5}  {'Command'}")
    print("-" * 60)

    arbiter = GimbalArbiter()
    state = ArbiterState(
        gimbal_state=GimbalState.IDLE,
        tracked_blobs=(),
        idle_duration_s=0.0,
        last_command_time=0.0,
        current_target_id=None,
    )

    # Simulation plan:
    #   Frames 1–10:  one blob with increasing persistence (1 → 10)
    #   Frames 11–20: no blobs — blob lost, should return to IDLE
    now = time.time()

    for frame_id in range(1, 21):
        if frame_id <= 10:
            # Ramp up persistence
            blob = make_blob(persistence_count=frame_id, blob_id=1)
            blobs: tuple[BlobMeta, ...] = (blob,)
        else:
            # Blob lost
            blobs = ()

        result = make_result(blobs=blobs, frame_id=frame_id)
        new_state, command, events = arbiter.step(state, result, now=now + frame_id)

        cmd_str = f"az={command.az_delta_deg:.1f}° el={command.el_delta_deg:.1f}°" if command else "—"
        n_blobs = len(blobs)
        print(f"{frame_id:>5}  {new_state.gimbal_state.value:15}  {n_blobs:>5}  {cmd_str}")

        # Print state transitions
        if new_state.gimbal_state != state.gimbal_state:
            print(f"        *** {state.gimbal_state.value} → {new_state.gimbal_state.value} ***")

        state = new_state

    print("-" * 60)
    print(f"Final state: {state.gimbal_state.value}")
    print("\nDemo complete.")
    print("Expected sequence: IDLE → ACQUIRING (frame 1) → TRACKING (frame 3) → IDLE (frame 11)")


if __name__ == "__main__":
    main()
