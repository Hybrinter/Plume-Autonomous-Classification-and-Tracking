"""Plume scene generation for SIL: synthetic raw mosaic frames + a scripted plume detector.

The scene renders radiometrically-plausible raw mosaic frames by compositing a Gaussian
plume signal over a uniform background, adding read-noise, quantizing to 12-bit uint16,
and interleaving into the 2x2 CFA mosaic via interleave_bands (the exact inverse of the
flight demosaic). This exercises the complete ingest path:
  calibrate_mosaic -> separate_bands -> normalize_dn -> select_bands -> compute_quality_flags.

The ScriptedDetector ignores the tensor content and detects from a fixed probability mask,
so a plume-rendered scene plus a plume mask yields a stable, strong off-boresight blob every
frame -- exactly what drives the gimbal arbiter to TRACKING.

Contains:
  - build_frames: N radiometrically-plausible (2048, 2448) uint16 MosaicFrame frames with
    monotonic frame_ids, deterministic for a given seed. The plume sits below boresight
    at band-plane (612, 900) so TRACKING commands negative elevation on the single axis.
  - plume_detector: a ScriptedDetector whose 1024x1224 mask yields one persistent blob at
    the full band-plane / inference tensor size (no crop, no scale).

Satisfies: REQ-AIML-IMAG-001, REQ-AIML-PREP-001.
"""

from __future__ import annotations

# third-party
import numpy as np

# internal
from flight.libs.types import MosaicFrame, Ok
from flight.payload.inference import ScriptedDetector
from flight.payload.preprocess import interleave_bands

MOSAIC_HEIGHT_PX = 2048  # along-track
MOSAIC_WIDTH_PX = 2448  # lateral
BAND_HEIGHT_PX = MOSAIC_HEIGHT_PX // 2  # 1024
BAND_WIDTH_PX = MOSAIC_WIDTH_PX // 2  # 1224
DETECTOR_HEIGHT_PX = BAND_HEIGHT_PX
DETECTOR_WIDTH_PX = BAND_WIDTH_PX
_BIT_DEPTH = 12
_FULL_SCALE = float(2**_BIT_DEPTH - 1)
# Background and plume amplitudes as fractions of full scale, per band plane in
# row-major cell order (BLUE, GREEN, RED, NIR). Smoke reflects strongest in NIR.
_BACKGROUND = (0.15, 0.15, 0.15, 0.18)
_PLUME_AMPLITUDE = (0.05, 0.08, 0.12, 0.25)
# Band-plane (x, y). Boresight is (612, 512); y=900 is below boresight -> -el.
_PLUME_X = 612.0
_PLUME_Y = 900.0
_PLUME_SIGMA = 40.0  # band-plane px
_NOISE_SIGMA_DN = 2.0


def build_frames(num_frames: int, seed: int = 0) -> list[MosaicFrame]:
    """Render num_frames raw mosaic frames: background + Gaussian plume + noise.

    Per band plane: dn = (background + amplitude * gaussian) * full_scale + noise,
    quantized to 12-bit uint16, then interleaved into the 2x2 CFA mosaic (the exact
    inverse of the flight demosaic). Deterministic for a given seed.

    Args:
        num_frames (int): Number of frames to generate.
        seed (int): NumPy random seed for deterministic noise (default 0).

    Returns:
        list[MosaicFrame]: num_frames frames, each a (2048, 2448) uint16 mosaic plane
        with frame_id running 1..num_frames and nominal exposure/gain metadata.
        NIR channel (plane 3) is brighter inside the plume region than the background,
        enabling the plume-brightness test.

    Notes:
        The Gaussian plume is centered at band-plane pixel (x=612, y=900) with sigma
        40 px: 388 px below the 1024x1224-plane boresight (612, 512). TRACKING issues
        a negative elevation RATE. Drivers pin azimuth at 0. Noise is i.i.d. Gaussian
        with sigma 2 DN, per-frame from the seeded RNG.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:BAND_HEIGHT_PX, 0:BAND_WIDTH_PX]
    gauss = np.exp(
        -(((yy - _PLUME_Y) ** 2 + (xx - _PLUME_X) ** 2) / (2.0 * _PLUME_SIGMA**2))
    ).astype(np.float32)  # np.ndarray[float32, (1024, 1224)]

    frames: list[MosaicFrame] = []
    for frame_id in range(1, num_frames + 1):
        signal = np.stack(
            [(_BACKGROUND[k] + _PLUME_AMPLITUDE[k] * gauss) * _FULL_SCALE for k in range(4)]
        ).astype(np.float32)  # np.ndarray[float32, (4, 1024, 1224)]
        noise = rng.normal(0.0, _NOISE_SIGMA_DN, size=signal.shape).astype(
            np.float32
        )  # np.ndarray[float32, (4, 1024, 1224)]
        planes = signal + noise  # np.ndarray[float32, (4, 1024, 1224)]
        mosaic_result = interleave_bands(planes)
        assert isinstance(mosaic_result, Ok)  # geometry is fixed; cannot fail
        mosaic = np.clip(mosaic_result.value, 0.0, _FULL_SCALE).astype(
            np.uint16
        )  # np.ndarray[uint16, (2048, 2448)]
        frames.append(
            MosaicFrame(
                timestamp_utc="2026-06-01T00:00:00.000Z",
                timestamp_s=float(frame_id),
                frame_id=frame_id,
                mosaic=mosaic,
                exposure_us=1000.0,
                gain_db=0.0,
            )
        )
    return frames


def plume_detector() -> ScriptedDetector:
    """Build a ScriptedDetector whose fixed mask yields one strong, stable off-center blob.

    Returns:
        ScriptedDetector: With a 50x50 unit-probability square (area 2500 px, confidence
        1.0) at tensor [875:925, 587:637] -- above the default gates. The mask is at
        full band-plane / inference resolution (1024 x 1224).

    Notes:
        The centroid (~611.5, ~899.5) sits ~388 px below boresight (612, 512). TRACKING
        issues a negative elevation RATE. Azimuth stays pinned at 0 in the drivers.
    """
    mask = np.zeros(
        (DETECTOR_HEIGHT_PX, DETECTOR_WIDTH_PX), dtype=np.float32
    )  # np.ndarray[float32, (H, W)]
    mask[875:925, 587:637] = 1.0  # centroid ~ (611.5, 899.5) in tensor / band-plane space
    return ScriptedDetector(mask, confidence_gate=0.55, min_blob_area_px=15)
