"""Plume scene generation for SIL: synthetic RGB frames + a scripted plume detector.

The scene renders three registered planes in ``BAND_ORDER`` (BLUE, GREEN, RED).
Flight then calibrates, normalizes, and cubic-upsamples them. The scripted mask
is at the upsampled inference size.

Contains:
  - build_frames: N (3, 1544, 2064) uint16 frames. The plume sits above boresight.
  - plume_detector: a ScriptedDetector whose mask matches the upsampled tensor.

Satisfies: REQ-AIML-IMAG-001, REQ-AIML-PREP-001.
"""

from __future__ import annotations

# third-party
import numpy as np

# internal
from flight.libs.types import BAND_ORDER, MosaicFrame
from flight.payload.inference import ScriptedDetector

PLANE_HEIGHT_PX = 1544
PLANE_WIDTH_PX = 2064
_UPSAMPLE = 2
DETECTOR_HEIGHT_PX = PLANE_HEIGHT_PX * _UPSAMPLE
DETECTOR_WIDTH_PX = PLANE_WIDTH_PX * _UPSAMPLE
_BIT_DEPTH = 12
_FULL_SCALE = float(2**_BIT_DEPTH - 1)
# BLUE, GREEN, RED. Red carries the strongest smoke contrast on this camera.
_BACKGROUND = (0.15, 0.15, 0.18)
_PLUME_AMPLITUDE = (0.05, 0.08, 0.20)
# Native (x, y). Boresight is (1032, 772). y=40 is above boresight -> +el.
_PLUME_X = 1032.0
_PLUME_Y = 40.0
_PLUME_SIGMA = 40.0
_NOISE_SIGMA_DN = 2.0


def build_frames(num_frames: int, seed: int = 0) -> list[MosaicFrame]:
    """Render num_frames RGB stacks: background + Gaussian plume + noise.

    Each plane is ``BAND_ORDER``. Values are 12-bit uint16. A given seed is
    deterministic.

    Args:
        num_frames (int): Number of frames to generate.
        seed (int): NumPy random seed for deterministic noise (default 0).

    Returns:
        list[MosaicFrame]: frames of shape (3, 1544, 2064). The red plane is
        brighter at the plume than in a far corner.

    Notes:
        The plume is centered at native pixel (x=1032, y=40), above boresight.
        TRACKING issues a positive elevation rate inside the science window.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:PLANE_HEIGHT_PX, 0:PLANE_WIDTH_PX]
    gauss = np.exp(
        -(((yy - _PLUME_Y) ** 2 + (xx - _PLUME_X) ** 2) / (2.0 * _PLUME_SIGMA**2))
    ).astype(np.float32)

    frames: list[MosaicFrame] = []
    for frame_id in range(1, num_frames + 1):
        signal = np.stack(
            [
                (_BACKGROUND[k] + _PLUME_AMPLITUDE[k] * gauss) * _FULL_SCALE
                for k in range(len(BAND_ORDER))
            ]
        ).astype(np.float32)
        noise = rng.normal(0.0, _NOISE_SIGMA_DN, size=signal.shape).astype(np.float32)
        planes = np.clip(signal + noise, 0.0, _FULL_SCALE).astype(np.uint16)
        frames.append(
            MosaicFrame(
                timestamp_utc="2026-06-01T00:00:00.000Z",
                timestamp_s=float(frame_id),
                frame_id=frame_id,
                planes=planes,
                exposure_us=1000.0,
                gain_db=0.0,
            )
        )
    return frames


def plume_detector() -> ScriptedDetector:
    """Build a ScriptedDetector whose fixed mask yields one strong, stable off-center blob.

    Returns:
        ScriptedDetector: With a 50x50 unit-probability square (area 2500 px, confidence
        1.0) at tensor [55:105, 2039:2089], above the default gates. The mask is at
        the upsampled inference resolution (3088 x 4128).

    Notes:
        The centroid (~2064, ~80) sits above boresight (2064, 1544). TRACKING issues
        a positive elevation rate inside the science window.
    """
    mask = np.zeros(
        (DETECTOR_HEIGHT_PX, DETECTOR_WIDTH_PX), dtype=np.float32
    )  # np.ndarray[float32, (H, W)]
    mask[55:105, 2039:2089] = 1.0  # centroid ~ (2064, 80) on the upsampled tensor
    return ScriptedDetector(mask, confidence_gate=0.55, min_blob_area_px=15)
