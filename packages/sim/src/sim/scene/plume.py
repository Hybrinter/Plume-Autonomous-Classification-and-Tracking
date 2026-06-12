"""Plume scene generation for SIL: synthetic raw mosaic frames + a scripted plume detector.

The scene renders radiometrically-plausible raw mosaic frames by compositing a Gaussian
plume signal over a uniform background, adding read-noise, quantizing to 12-bit uint16,
and interleaving into the 2x2 CFA mosaic via interleave_bands (the exact inverse of the
flight demosaic). This exercises the complete ingest path:
  calibrate_mosaic -> separate_bands -> normalize_dn -> select_bands -> compute_quality_flags.

The ScriptedDetector ignores the tensor content and detects from a fixed probability mask,
so a plume-rendered scene plus a plume mask yields a stable, strong central blob every
frame -- exactly what drives the gimbal arbiter to TRACKING.

Contains:
  - build_frames: N radiometrically-plausible (1024, 1024) uint16 MosaicFrame frames with
    monotonic frame_ids, deterministic for a given seed. The plume sits off-center at
    band-plane (340, 340) so its boresight displacement drives TRACKING commands.
  - plume_detector: a ScriptedDetector whose 256x256 mask yields one persistent blob (the
    mask is at inference-tensor resolution: the 512 band plane decimated 2x in search
    mode, where the plume appears at tensor ~(170, 170)).

Satisfies: REQ-AIML-IMAG-001, REQ-AIML-PREP-001.
"""

from __future__ import annotations

# third-party
import numpy as np

# internal
from flight.libs.types import MosaicFrame, Ok
from flight.payload.model import ScriptedDetector
from flight.payload.preprocess import interleave_bands

FRAME_SIZE = 1024  # mosaic plane size; band planes are 512x512
DETECTOR_SIZE = 256  # inference tensor size; the 512 band plane decimated 2x in search mode
_BIT_DEPTH = 12
_FULL_SCALE = float(2**_BIT_DEPTH - 1)
# Background and plume amplitudes as fractions of full scale, per band plane in
# row-major cell order (BLUE, GREEN, RED, NIR). Smoke reflects strongest in NIR.
_BACKGROUND = (0.15, 0.15, 0.15, 0.18)
_PLUME_AMPLITUDE = (0.05, 0.08, 0.12, 0.25)
_PLUME_CENTER = (340.0, 340.0)  # band-plane px; ~119 px off the (256, 256) boresight
_PLUME_SIGMA = 24.0  # band-plane px
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
        list[MosaicFrame]: num_frames frames, each a (1024, 1024) uint16 mosaic plane
        with frame_id running 1..num_frames and nominal exposure/gain metadata.
        NIR channel (plane 3) is brighter inside the plume region than the background,
        enabling the plume-brightness test.

    Notes:
        The Gaussian plume is centered at band-plane pixel (340, 340) with sigma 24 px:
        ~119 px off the 512-plane boresight (256, 256), above the minimum deadband and
        below the maximum, so TRACKING commands flow. In decimated search mode it appears
        at tensor ~(170, 170), inside the scripted mask region [145:195, 145:195]. Noise
        is i.i.d. Gaussian with sigma 2 DN, per-frame from the seeded RNG.
    """
    rng = np.random.default_rng(seed)
    half = FRAME_SIZE // 2
    yy, xx = np.mgrid[0:half, 0:half]  # np.ndarray[int, (512, 512)] each
    gauss = np.exp(
        -(((yy - _PLUME_CENTER[0]) ** 2 + (xx - _PLUME_CENTER[1]) ** 2) / (2.0 * _PLUME_SIGMA**2))
    ).astype(np.float32)  # np.ndarray[float32, (512, 512)]

    frames: list[MosaicFrame] = []
    for frame_id in range(1, num_frames + 1):
        signal = np.stack(
            [(_BACKGROUND[k] + _PLUME_AMPLITUDE[k] * gauss) * _FULL_SCALE for k in range(4)]
        ).astype(np.float32)  # np.ndarray[float32, (4, 512, 512)]
        noise = rng.normal(0.0, _NOISE_SIGMA_DN, size=signal.shape).astype(
            np.float32
        )  # np.ndarray[float32, (4, 512, 512)]
        planes = signal + noise  # np.ndarray[float32, (4, 512, 512)]
        mosaic_result = interleave_bands(planes)
        assert isinstance(mosaic_result, Ok)  # geometry is fixed; cannot fail
        mosaic = np.clip(mosaic_result.value, 0.0, _FULL_SCALE).astype(
            np.uint16
        )  # np.ndarray[uint16, (1024, 1024)]
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
    """Build a ScriptedDetector whose fixed mask yields one strong, stable off-center blob.

    Returns:
        ScriptedDetector: With a 50x50 unit-probability square (area 2500 px, confidence
        1.0) at tensor [145:195, 145:195] -- above the default gates. The mask is at
        inference-tensor resolution (256), matching the scene plume's decimated
        search-mode position.

    Notes:
        The centroid (~169.5, ~169.5) back-projects in search mode (scale 0.5) to
        band-plane (~339, ~339): ~117 px off the 512-plane boresight, between the minimum
        and maximum deadbands, letting TRACKING issue RATE commands that move the gimbal
        off the origin in the closed loop. In TRACKING ROI mode (scale 1.0, crop clamped
        at the plane edge) the displacement stays below the maximum deadband.
    """
    mask = np.zeros((DETECTOR_SIZE, DETECTOR_SIZE), dtype=np.float32)  # np.ndarray[float32, (H, W)]
    mask[145:195, 145:195] = 1.0  # centroid ~ (169.5, 169.5) in tensor space
    return ScriptedDetector(mask, confidence_gate=0.55, min_blob_area_px=15)
