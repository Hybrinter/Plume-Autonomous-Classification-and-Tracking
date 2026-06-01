"""
pact.preprocessing.quality -- Per-frame quality flag computation for PACT inference gating.

Satisfies: REQ-AIML-IMAG-002, REQ-AIML-DATA-003

Computes a frozenset of FrameUsabilityTag flags for each calibrated frame. These flags
are attached to ProcessedFrameMsg and used downstream to decide whether to run inference
and how to classify the frame for dataset curation (training vs. tracking vs. invalid).

Flag conditions (all thresholds marked TODO: move to config):
    SATURATED           -- any band has > 5% of pixels above 0.95 (post-normalisation)
    MOTION_SMEAR        -- placeholder; raised when exposure time implies smear at ISS speed
    CLOUD_CONTAMINATED  -- NIR/Red ratio heuristic exceeds threshold
    SUNGLINT            -- mean NIR band intensity exceeds threshold

Band index assumptions (for a (4, H, W) array after select_bands):
    index 0 -> B2 (blue)
    index 1 -> B3 (green)
    index 2 -> B4 (red)
    index 3 -> B8 (NIR)
"""

from __future__ import annotations

# stdlib
from typing import Final

# third-party
import numpy as np

# internal
from flight.libs.config import PreprocessingConfig
from flight.libs.types import FrameUsabilityTag

# Saturation pixel level is a fixed normalisation constant, not a tunable threshold.
SATURATION_PIXEL_LEVEL: Final[float] = 0.95  # normalised DN units


def compute_quality_flags(
    bands: object,  # np.ndarray[float32, (C, H, W)]
    exposure_us: float,
    utc_timestamp: str,
    cfg: PreprocessingConfig,
) -> frozenset[FrameUsabilityTag]:
    """Compute per-frame quality flags for a calibrated multispectral frame.

    Evaluates five independent heuristic conditions and returns the set of
    flags that are raised. An empty frozenset means the frame is clean.

    Args:
        bands:         Calibrated and normalised band array, shape (C, H, W),
                       float32. C >= 4, band ordering [B2, B3, B4, B8].
        exposure_us:   Camera exposure time in microseconds.
        utc_timestamp: ISO 8601 timestamp string from the frame metadata.
        cfg:           PreprocessingConfig with quality-flag thresholds.

    Returns:
        frozenset of FrameUsabilityTag flags raised for this frame.
        An empty frozenset indicates a clean, inference-ready frame.
    """
    flags: set[FrameUsabilityTag] = set()

    # --- INCOMPLETE_METADATA ---
    if exposure_us <= 0 or not utc_timestamp:
        flags.add(FrameUsabilityTag.INCOMPLETE_METADATA)

    # --- SATURATED ---
    # Check each band independently; flag if any band exceeds the threshold.
    bands_arr: np.ndarray = bands  # type: ignore[assignment]
    n_pixels: int = bands_arr.shape[1] * bands_arr.shape[2]
    for c in range(bands_arr.shape[0]):
        saturated_count: int = int((bands_arr[c] > SATURATION_PIXEL_LEVEL).sum())
        if saturated_count / n_pixels > cfg.saturation_fraction_threshold:
            flags.add(FrameUsabilityTag.SATURATED)
            break

    # --- MOTION_SMEAR (placeholder) ---
    if exposure_us > cfg.motion_smear_exposure_us:
        flags.add(FrameUsabilityTag.MOTION_SMEAR)

    # --- CLOUD_CONTAMINATED ---
    # Band index 2 = B4 (Red), index 3 = B8 (NIR).
    red_band: np.ndarray = bands_arr[2]  # np.ndarray[float32, (H, W)]
    nir_band: np.ndarray = bands_arr[3]  # np.ndarray[float32, (H, W)]
    epsilon: float = 1e-6
    nir_red_ratio: float = float(nir_band.mean()) / (float(red_band.mean()) + epsilon)
    if nir_red_ratio > cfg.nir_red_ratio_threshold:
        flags.add(FrameUsabilityTag.CLOUD_CONTAMINATED)

    # --- SUNGLINT ---
    nir_mean: float = float(nir_band.mean())
    if nir_mean > cfg.sunglint_nir_mean_threshold:
        flags.add(FrameUsabilityTag.SUNGLINT)

    return frozenset(flags)
