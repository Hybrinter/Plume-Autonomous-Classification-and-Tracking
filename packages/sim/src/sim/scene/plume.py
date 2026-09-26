"""Plume scene generation for SIL: synthetic prism frames + a scripted plume detector.

The scene renders three registered planes by compositing a Gaussian plume signal
over a uniform background, adding read-noise, and quantizing to 12-bit uint16.
This exercises the complete ingest path:
  stack_channels -> calibrate_mosaic -> normalize_dn -> select_bands -> compute_quality_flags.

The ScriptedDetector ignores the tensor content and detects from a fixed probability mask,
so a plume-rendered scene plus a plume mask yields a stable, strong off-boresight blob every
frame -- exactly what drives the gimbal arbiter to TRACKING.

Contains:
  - build_frames: N radiometrically-plausible (3, 1544, 2064) uint16 MosaicFrame buffers
    with monotonic frame_ids, deterministic for a given seed. The plume sits the same
    fraction off boresight as the previous (612, 124) mark on a 1224 x 1024 plane.
    Boresight is (1032, 772).
  - plume_detector: a ScriptedDetector whose mask yields one persistent blob at the
    full frame / inference tensor size (no crop, no scale).

Satisfies: REQ-AIML-IMAG-001, REQ-AIML-PREP-001.
"""

from __future__ import annotations

# third-party
import numpy as np

# internal
from flight.libs.types import MosaicFrame
from flight.payload.inference import ScriptedDetector

FRAME_HEIGHT_PX = 1544  # along-track
FRAME_WIDTH_PX = 2064  # lateral
# Old band plane was 1224 x 1024 with the plume at (612, 124) and boresight (612, 512).
_OLD_HEIGHT_PX = 1024.0
_OLD_WIDTH_PX = 1224.0
_OLD_PLUME_X = 612.0
_OLD_PLUME_Y = 124.0
_PLUME_X = FRAME_WIDTH_PX / 2.0 + (_OLD_PLUME_X - _OLD_WIDTH_PX / 2.0) * (
    FRAME_WIDTH_PX / _OLD_WIDTH_PX
)
_PLUME_Y = FRAME_HEIGHT_PX / 2.0 + (_OLD_PLUME_Y - _OLD_HEIGHT_PX / 2.0) * (
    FRAME_HEIGHT_PX / _OLD_HEIGHT_PX
)
_PLUME_SIGMA = 40.0 * (FRAME_HEIGHT_PX / _OLD_HEIGHT_PX)
_BIT_DEPTH = 12
_FULL_SCALE = float(2**_BIT_DEPTH - 1)
# Background and plume amplitudes as fractions of full scale, wire order RED, GREEN, BLUE.
_BACKGROUND = (0.15, 0.15, 0.15)
_PLUME_AMPLITUDE = (0.12, 0.08, 0.05)
_NOISE_SIGMA_DN = 2.0
# 50 px box on the old plane, scaled onto the new frame and centered on the plume.
_MASK_HALF_X = 25.0 * (FRAME_WIDTH_PX / _OLD_WIDTH_PX)
_MASK_HALF_Y = 25.0 * (FRAME_HEIGHT_PX / _OLD_HEIGHT_PX)


def build_frames(num_frames: int, seed: int = 0) -> list[MosaicFrame]:
    """Render num_frames prism buffers: background + Gaussian plume + noise.

    Per channel: dn = (background + amplitude * gaussian) * full_scale + noise,
    quantized to 12-bit uint16 and stacked as (3, H, W) in RED, GREEN, BLUE order.
    Deterministic for a given seed.

    Args:
        num_frames (int): Number of frames to generate.
        seed (int): NumPy random seed for deterministic noise (default 0).

    Returns:
        list[MosaicFrame]: num_frames frames, each a (3, 1544, 2064) uint16 buffer
        with frame_id running 1..num_frames and nominal exposure/gain metadata.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:FRAME_HEIGHT_PX, 0:FRAME_WIDTH_PX]
    gauss = np.exp(
        -(((yy - _PLUME_Y) ** 2 + (xx - _PLUME_X) ** 2) / (2.0 * _PLUME_SIGMA**2))
    ).astype(np.float32)

    frames: list[MosaicFrame] = []
    for frame_id in range(1, num_frames + 1):
        signal = np.stack(
            [(_BACKGROUND[k] + _PLUME_AMPLITUDE[k] * gauss) * _FULL_SCALE for k in range(3)]
        ).astype(np.float32)
        noise = rng.normal(0.0, _NOISE_SIGMA_DN, size=signal.shape).astype(np.float32)
        planes = np.clip(signal + noise, 0.0, _FULL_SCALE).astype(np.uint16)
        frames.append(
            MosaicFrame(
                timestamp_utc="2026-06-01T00:00:00.000Z",
                timestamp_s=float(frame_id),
                frame_id=frame_id,
                mosaic=planes,
                exposure_us=1000.0,
                gain_db=0.0,
            )
        )
    return frames


def plume_detector() -> ScriptedDetector:
    """Build a ScriptedDetector whose fixed mask yields one strong, stable off-center blob.

    Returns:
        ScriptedDetector: With a unit-probability rectangle (confidence 1.0) above
        the default gates. The mask is at full frame / inference resolution
        (1544 x 2064). The rectangle is the old 50 px box scaled onto this frame
        and centered on the plume.
    """
    mask = np.zeros((FRAME_HEIGHT_PX, FRAME_WIDTH_PX), dtype=np.float32)
    y0 = int(round(_PLUME_Y - _MASK_HALF_Y))
    y1 = int(round(_PLUME_Y + _MASK_HALF_Y))
    x0 = int(round(_PLUME_X - _MASK_HALF_X))
    x1 = int(round(_PLUME_X + _MASK_HALF_X))
    mask[y0:y1, x0:x1] = 1.0
    return ScriptedDetector(mask, confidence_gate=0.55, min_blob_area_px=15)
