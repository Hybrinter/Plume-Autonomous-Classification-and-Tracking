"""
pact.preprocessing.quality — Per-frame quality flag computation for PACT inference gating.

Satisfies: REQ-AIML-IMAG-002, REQ-AIML-DATA-003

Computes a frozenset of FrameUsabilityTag flags for each calibrated frame. These flags
are attached to ProcessedFrameMsg and used downstream to decide whether to run inference
and how to classify the frame for dataset curation (training vs. tracking vs. invalid).

Flag conditions (all thresholds marked TODO: move to config):
    SATURATED           — any band has > 5% of pixels above 0.95 (post-normalisation)
    MOTION_SMEAR        — placeholder; raised when exposure time implies smear at ISS speed
    CLOUD_CONTAMINATED  — NIR/Red ratio heuristic exceeds threshold
    SUNGLINT            — mean NIR band intensity exceeds threshold

Band index assumptions (for a (4, H, W) array after select_bands):
    index 0 → B2 (blue)
    index 1 → B3 (green)
    index 2 → B4 (red)
    index 3 → B8 (NIR)
"""

from __future__ import annotations

# stdlib
from typing import Final

# third-party
import numpy as np

# internal
from pact.types.enums import FrameUsabilityTag


# ---------------------------------------------------------------------------
# Threshold constants
# All values below are placeholders pending empirical tuning from on-orbit data.
# TODO: move all thresholds to PactConfig / config/default.toml once a
#       PreprocessingConfig dataclass is introduced in pact/types/config.py.
# ---------------------------------------------------------------------------

# SATURATED: fraction of pixels in any single band that may exceed the saturation
# level before the frame is flagged. Value: 5% of pixels above 0.95 (normalised).
SATURATION_FRACTION_THRESHOLD: Final[float] = 0.05
SATURATION_PIXEL_LEVEL: Final[float] = 0.95  # normalised DN units

# CLOUD_CONTAMINATED: NIR-to-Red ratio threshold. Clouds have high NIR and high Red
# reflectance; the ratio is used as a coarse heuristic until a proper cloud mask is
# available. Threshold derived from Sentinel-2 L2A literature (approximate).
# TODO: tune from real on-orbit imagery.
NIR_RED_RATIO_THRESHOLD: Final[float] = 3.0

# SUNGLINT: mean NIR band intensity above which sunglint is suspected.
# Sunglint causes artificially high NIR values over water surfaces.
# TODO: tune from real on-orbit imagery.
SUNGLINT_NIR_MEAN_THRESHOLD: Final[float] = 0.6

# MOTION_SMEAR: ISS orbital velocity is ~7.66 km/s at ~420 km altitude.
# At ground sampling distance ~10 m (Sentinel-2) and 30 fps, one frame covers ~255 m.
# If exposure_us is long enough that the PSF smear exceeds 0.5 pixels, flag the frame.
# Currently a rough heuristic — proper implementation requires gimbal slew telemetry.
# TODO: replace with actual gimbal slew rate from GimbalCommandMsg telemetry.
MOTION_SMEAR_EXPOSURE_THRESHOLD_US: Final[float] = 5000.0  # microseconds


def compute_quality_flags(
    bands: np.ndarray,       # (C, H, W) float32, calibrated and normalised to [0, 1]
    exposure_us: float,
    gain_db: float,
) -> frozenset[FrameUsabilityTag]:
    """Compute per-frame quality flags for a calibrated multispectral frame.

    Evaluates four independent heuristic conditions and returns the set of flags that
    are raised. An empty frozenset means the frame is clean.

    Flag conditions:

    SATURATED:
        Raised if any band contains more than SATURATION_FRACTION_THRESHOLD (5%) of
        pixels above SATURATION_PIXEL_LEVEL (0.95). Saturation causes clipping artefacts
        that corrupt plume boundary detection.

    MOTION_SMEAR:
        Placeholder — raised if exposure_us exceeds MOTION_SMEAR_EXPOSURE_THRESHOLD_US.
        This is a conservative proxy for actual motion blur estimation, which requires
        gimbal slew rate telemetry not yet available in this pipeline stage.
        TODO: replace with gimbal slew rate check when GimbalCommandMsg is fed into
        preprocessing.

    CLOUD_CONTAMINATED:
        Raised if the mean (NIR / Red) ratio across the frame exceeds
        NIR_RED_RATIO_THRESHOLD. Cloud pixels have high reflectance in both bands;
        a high ratio indicates pervasive bright scattering that obscures plumes.
        # TODO: tune threshold from real on-orbit cloud-over-plume imagery.

    SUNGLINT:
        Raised if the mean NIR band intensity exceeds SUNGLINT_NIR_MEAN_THRESHOLD.
        Over-ocean sunglint produces anomalously high NIR reflectance.
        # TODO: tune threshold from real on-orbit ocean-sunglint imagery.

    Args:
        bands:       Calibrated and normalised band array, shape (C, H, W), float32.
                     C >= 4, with band ordering [B2, B3, B4, B8] at indices [0,1,2,3].
        exposure_us: Camera exposure time in microseconds at frame capture time.
        gain_db:     Camera analogue gain in dB at frame capture time (unused by current
                     heuristics but retained for future use).

    Returns:
        frozenset of FrameUsabilityTag flags that are raised for this frame.
        An empty frozenset indicates a clean, inference-ready frame.
    """
    flags: set[FrameUsabilityTag] = set()

    # --- SATURATED ---
    # Check each band independently; flag if any band exceeds the threshold.
    n_pixels: int = bands.shape[1] * bands.shape[2]
    for c in range(bands.shape[0]):
        saturated_count: int = int((bands[c] > SATURATION_PIXEL_LEVEL).sum())
        if saturated_count / n_pixels > SATURATION_FRACTION_THRESHOLD:
            flags.add(FrameUsabilityTag.SATURATED)
            break  # one band is enough to flag; no need to check remaining bands

    # --- MOTION_SMEAR (placeholder) ---
    # TODO: replace with actual gimbal slew rate when telemetry is available.
    if exposure_us > MOTION_SMEAR_EXPOSURE_THRESHOLD_US:
        flags.add(FrameUsabilityTag.MOTION_SMEAR)

    # --- CLOUD_CONTAMINATED ---
    # Band index 2 = B4 (Red), index 3 = B8 (NIR).
    # Add a small epsilon to the Red band to avoid division by zero over dark pixels.
    red_band: np.ndarray = bands[2]   # np.ndarray[float32, (H, W)]
    nir_band: np.ndarray = bands[3]   # np.ndarray[float32, (H, W)]
    epsilon: float = 1e-6
    nir_red_ratio: float = float(nir_band.mean()) / (float(red_band.mean()) + epsilon)
    if nir_red_ratio > NIR_RED_RATIO_THRESHOLD:
        # TODO: tune NIR_RED_RATIO_THRESHOLD from real on-orbit cloud imagery.
        flags.add(FrameUsabilityTag.CLOUD_CONTAMINATED)

    # --- SUNGLINT ---
    nir_mean: float = float(nir_band.mean())
    if nir_mean > SUNGLINT_NIR_MEAN_THRESHOLD:
        # TODO: tune SUNGLINT_NIR_MEAN_THRESHOLD from real on-orbit ocean imagery.
        flags.add(FrameUsabilityTag.SUNGLINT)

    return frozenset(flags)
