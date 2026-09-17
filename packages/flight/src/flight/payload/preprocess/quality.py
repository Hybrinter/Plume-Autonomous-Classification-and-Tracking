"""flight.payload.preprocess.quality -- Per-frame quality flag computation for inference gating.

Satisfies: REQ-AIML-IMAG-002, REQ-AIML-DATA-003

Computes a frozenset of FrameUsabilityTag flags for each calibrated, normalized frame.
These flags are attached to ProcessedFrameMsg and used downstream to decide whether to
run inference and how to classify the frame for dataset curation (training vs. tracking
vs. invalid).

Flag conditions:
    SATURATED           -- any band has > saturation_fraction_threshold of pixels above
                           SATURATION_PIXEL_LEVEL (post-normalisation).
    CLOUD_CONTAMINATED  -- NIR/Red mean ratio exceeds cfg.nir_red_ratio_threshold.
    SUNGLINT            -- mean NIR band intensity exceeds cfg.sunglint_nir_mean_threshold.
    INCOMPLETE_METADATA -- nonpositive exposure or missing timestamp.

MOTION_SMEAR is not raised. Along-track smear is a control cap in outer_rate
(max_motion_smear_px). FAST_REWIND smears on purpose; exclude those frames by
gimbal mode, not a second smear inequality.

Band index assumptions (for a (C, H, W) array after select_bands), order
[BLUE, GREEN, RED, NIR]:
    index 0 -> BLUE
    index 1 -> GREEN
    index 2 -> RED
    index 3 -> NIR

Contains:
  - SmearRateSource: provenance of the elevation rate the payload app selected.
  - compute_quality_flags: evaluate the heuristics and return the raised-flag frozenset.
"""

from __future__ import annotations

# stdlib
from enum import Enum
from typing import Final

# third-party
import numpy as np

# internal
from flight.libs.config import PreprocessingConfig
from flight.libs.types import FrameUsabilityTag

# Saturation pixel level is a fixed normalisation constant, not a tunable threshold.
SATURATION_PIXEL_LEVEL: Final[float] = 0.95  # normalised DN units


class SmearRateSource(Enum):
    """Provenance of the elevation rate selected by the payload app.

    String values mirror member names. ``0.0`` is a valid measured, encoder, or
    commanded rate. Unknown measured plus a failed encoder bracket falls back
    to the commanded rate and labels it COMMANDED. Quality flags do not use
    this rate to raise MOTION_SMEAR.
    """

    MEASURED = "MEASURED"
    ENCODER = "ENCODER"
    COMMANDED = "COMMANDED"


def compute_quality_flags(
    bands: object,  # np.ndarray[float32, (C, H, W)], order [BLUE, GREEN, RED, NIR]
    exposure_us: float,
    slew_rate_deg_per_s: float,
    ifov_band_deg_per_px: float,
    utc_timestamp: str,
    cfg: PreprocessingConfig,
    omega_scene_el_deg_per_s: float = 0.0,
) -> frozenset[FrameUsabilityTag]:
    """Compute per-frame quality flags for a calibrated, normalized multispectral frame.

    Evaluates independent heuristic conditions and returns the set of flags raised. An
    empty frozenset means the frame is clean and inference-ready.

    Inputs:
        bands (np.ndarray[float32, (C, H, W)]): Calibrated and normalised band array.
            C >= 4, band ordering [BLUE, GREEN, RED, NIR].
        exposure_us (float): Camera exposure time in microseconds.
        slew_rate_deg_per_s (float): Gimbal elevation rate in degrees per second
            over the exposure. ``0.0`` is a stationary gimbal. Unused for flags;
            smear is a control cap, not a keep/drop.
        ifov_band_deg_per_px (float): Instantaneous field of view per band-plane pixel,
            degrees per pixel (SensorConfig.ifov_band_deg_per_px). Unused for flags.
        utc_timestamp (str): ISO 8601 timestamp string from the frame metadata.
        cfg (PreprocessingConfig): Quality-flag thresholds.
        omega_scene_el_deg_per_s (float): Nominal scene elevation rate in degrees per
            second. Unused for flags.

    Outputs:
        frozenset[FrameUsabilityTag]: The flags raised for this frame; empty if clean.

    Notes:
        MOTION_SMEAR is not raised. Dataset exclusion for hardware-slew hunt frames
        is gimbal_state FAST_REWIND.
    """
    del slew_rate_deg_per_s, ifov_band_deg_per_px, omega_scene_el_deg_per_s
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

    # --- CLOUD_CONTAMINATED ---
    # Band index 2 = RED, index 3 = NIR.
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
