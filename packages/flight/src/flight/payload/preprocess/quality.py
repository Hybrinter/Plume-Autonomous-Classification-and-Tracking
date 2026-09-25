"""flight.payload.preprocess.quality -- Per-frame quality flag computation for inference gating.

Satisfies: REQ-AIML-IMAG-002, REQ-AIML-DATA-003

Computes a frozenset of FrameUsabilityTag flags for each calibrated, normalized frame.
These flags are attached to ProcessedFrameMsg and used downstream to decide whether to
run inference and how to classify the frame for dataset curation (training vs. tracking
vs. invalid).

Flag conditions:
    SATURATED           -- any channel has > saturation_fraction_threshold of pixels above
                           SATURATION_PIXEL_LEVEL (post-normalisation).
    MOTION_SMEAR        -- physical: elevation-relative smear length in pixels,
                           smear_px = abs(slew_rate_deg_per_s - omega_scene_el_deg_per_s)
                           * (exposure_us * 1e-6) / IFOV, exceeds cfg.max_motion_smear_px.
                           Azimuth motion does not contribute.
    INCOMPLETE_METADATA -- nonpositive exposure or missing timestamp.

Bands are a (C, H, W) array after select_bands. Channel order follows
InferenceConfig.input_bands. Saturation checks every channel.

Contains:
  - SmearRateSource: provenance of the elevation rate used for MOTION_SMEAR.
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
    """Provenance of the elevation rate used for MOTION_SMEAR.

    String values mirror member names. ``0.0`` is a valid measured, encoder, or
    commanded rate. Unknown measured plus a failed encoder bracket falls back
    to the commanded rate and labels it COMMANDED.
    """

    MEASURED = "MEASURED"
    ENCODER = "ENCODER"
    COMMANDED = "COMMANDED"


def compute_quality_flags(
    bands: object,  # np.ndarray[float32, (C, H, W)]
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
        bands (np.ndarray[float32, (C, H, W)]): Calibrated and normalised channel array.
        exposure_us (float): Camera exposure time in microseconds.
        slew_rate_deg_per_s (float): Gimbal elevation rate in degrees per second
            over the exposure. ``0.0`` is a stationary gimbal.
        ifov_band_deg_per_px (float): Instantaneous field of view per pixel,
            degrees per pixel (SensorOpticsConfig.ifov_band_deg_per_px).
        utc_timestamp (str): ISO 8601 timestamp string from the frame metadata.
        cfg (PreprocessingConfig): Quality-flag thresholds.
        omega_scene_el_deg_per_s (float): Nominal scene elevation rate in degrees per
            second (co-rotation / tracking feedforward). Defaults to 0.0 when unknown.

    Outputs:
        frozenset[FrameUsabilityTag]: The flags raised for this frame; empty if clean.

    Notes:
        MOTION_SMEAR uses elevation-relative blur: the mismatch between gimbal and scene
        elevation rates during the exposure, converted to band-plane pixels via the IFOV.
        Matching rates, including both zero, do not flag. Azimuth motion is not modeled
        and cannot raise MOTION_SMEAR.
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

    # --- MOTION_SMEAR: elevation-relative smear length in band-plane pixels ---
    rel_rate_deg_per_s = abs(slew_rate_deg_per_s - omega_scene_el_deg_per_s)
    smear_px = rel_rate_deg_per_s * (exposure_us * 1e-6) / ifov_band_deg_per_px
    if smear_px > cfg.max_motion_smear_px:
        flags.add(FrameUsabilityTag.MOTION_SMEAR)

    return frozenset(flags)
