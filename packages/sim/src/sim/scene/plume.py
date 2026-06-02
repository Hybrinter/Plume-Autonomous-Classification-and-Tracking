"""Plume scene generation for SIL: synthetic raw frames + a scripted plume detector.

The frames are zeroed (4, 256, 256) band stacks matching the payload's identity
calibration shape; the ScriptedDetector ignores the tensor content and detects from a
fixed probability mask, so a zeroed scene plus a plume mask yields a stable, strong
central blob every frame -- exactly what drives the gimbal arbiter to TRACKING.

Contains:
  - build_frames: N zeroed RawFrameMsg frames with monotonic frame_ids.
  - plume_detector: a ScriptedDetector whose mask yields one persistent central blob.
"""

from __future__ import annotations

# third-party
import numpy as np

# internal
from flight.libs.messages import RawFrameMsg
from flight.libs.types import MessageType
from flight.payload.model import ScriptedDetector

FRAME_BANDS = 4
FRAME_SIZE = 256


def build_frames(num_frames: int) -> list[RawFrameMsg]:
    """Build a list of zeroed raw frames for the SIL sensor to replay.

    Args:
        num_frames: Number of frames to generate.

    Returns:
        A list of num_frames RawFrameMsg, each a zeroed (4, 256, 256) float32 band
        stack with frame_id running 1..num_frames.
    """
    frames: list[RawFrameMsg] = []
    for frame_id in range(1, num_frames + 1):
        raw_bands = np.zeros(
            (FRAME_BANDS, FRAME_SIZE, FRAME_SIZE), dtype=np.float32
        )  # np.ndarray[float32, (C, H, W)]
        frames.append(
            RawFrameMsg(
                msg_type=MessageType.RAW_FRAME,
                timestamp_utc="2026-06-01T00:00:00.000Z",
                frame_id=frame_id,
                raw_bands=raw_bands,
                exposure_us=1000.0,
                gain_db=0.0,
                gimbal_az_deg=0.0,
                gimbal_el_deg=0.0,
            )
        )
    return frames


def plume_detector() -> ScriptedDetector:
    """Build a ScriptedDetector whose fixed mask yields one strong, stable central blob.

    Returns:
        A ScriptedDetector with a 50x50 unit-probability square (area 2500 px,
        confidence 1.0) centered in a 256x256 mask -- above the default gates.
    """
    mask = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=np.float32)  # np.ndarray[float32, (H, W)]
    mask[100:150, 100:150] = 1.0
    return ScriptedDetector(mask, confidence_gate=0.55, min_blob_area_px=15)
