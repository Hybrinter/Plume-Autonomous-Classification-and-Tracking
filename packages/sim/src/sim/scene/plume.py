"""Plume scene generation for SIL: synthetic raw mosaic frames + a scripted plume detector.

The frames are zeroed (512, 512) uint16 mosaic planes matching the sensor geometry; the
ScriptedDetector ignores the tensor content and detects from a fixed probability mask, so
a zeroed scene plus a plume mask yields a stable, strong central blob every frame --
exactly what drives the gimbal arbiter to TRACKING. (Radiometrically-plausible rendering
through the full ingest path lands in a later task; this scene exercises the wiring.)

Contains:
  - build_frames: N zeroed (512, 512) uint16 MosaicFrame frames with monotonic frame_ids.
  - plume_detector: a ScriptedDetector whose 256x256 mask yields one persistent central
    blob (the mask is already at band-plane resolution -- sensor size / 2).
"""

from __future__ import annotations

# third-party
import numpy as np

# internal
from flight.libs.types import MosaicFrame
from flight.payload.model import ScriptedDetector

FRAME_SIZE = 512  # mosaic plane size; band planes are 256x256
DETECTOR_SIZE = 256  # band-plane (post-demosaic) size; the scripted mask matches this


def build_frames(num_frames: int) -> list[MosaicFrame]:
    """Build a list of zeroed raw mosaic frames for the SIL sensor to replay.

    Inputs:
        num_frames (int): Number of frames to generate.

    Outputs:
        list[MosaicFrame]: num_frames frames, each a zeroed (512, 512) uint16 mosaic
        plane with frame_id running 1..num_frames and nominal exposure/gain metadata.
    """
    frames: list[MosaicFrame] = []
    for frame_id in range(1, num_frames + 1):
        mosaic = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=np.uint16)  # np.ndarray[uint16, (H, W)]
        frames.append(
            MosaicFrame(
                timestamp_utc="2026-06-01T00:00:00.000Z",
                frame_id=frame_id,
                mosaic=mosaic,
                exposure_us=1000.0,
                gain_db=0.0,
            )
        )
    return frames


def plume_detector() -> ScriptedDetector:
    """Build a ScriptedDetector whose fixed mask yields one strong, stable central blob.

    Inputs:
        None.

    Outputs:
        ScriptedDetector: With a 50x50 unit-probability square (area 2500 px, confidence
        1.0) centered in a 256x256 mask -- above the default gates. The mask resolution
        matches the demosaicked band planes (sensor size / 2).
    """
    mask = np.zeros((DETECTOR_SIZE, DETECTOR_SIZE), dtype=np.float32)  # np.ndarray[float32, (H, W)]
    mask[100:150, 100:150] = 1.0
    return ScriptedDetector(mask, confidence_gate=0.55, min_blob_area_px=15)
