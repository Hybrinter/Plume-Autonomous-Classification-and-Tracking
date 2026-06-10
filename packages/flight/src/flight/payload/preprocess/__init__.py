"""Payload preprocessing: pure functions transforming raw bands for inference.

Stage order in the payload loop (mosaic path): calibrate_mosaic -> separate_bands ->
normalize_dn -> select_bands -> compute_quality_flags -> crop. All functions are pure
(no I/O, no global state).

Legacy (C, H, W) path: RadiometricCalibration + apply_calibration (retained until
the ingest switchover in Task 7 is complete; will be removed after Task 7).

Satisfies: REQ-AIML-PREP-001, REQ-AIML-PREP-002, REQ-AIML-IMAG-001.
"""

from flight.payload.preprocess.band_select import BAND_INDICES, select_bands
from flight.payload.preprocess.crop import backproject_pixel, crop_to_roi
from flight.payload.preprocess.demosaic import CELL_OFFSETS, interleave_bands, separate_bands
from flight.payload.preprocess.normalize import normalize_dn
from flight.payload.preprocess.quality import compute_quality_flags
from flight.payload.preprocess.radiometric import (
    MosaicCalibration,
    RadiometricCalibration,
    apply_calibration,
    calibrate_mosaic,
    correct_bad_pixels,
)

__all__ = [
    "BAND_INDICES",
    "CELL_OFFSETS",
    "MosaicCalibration",
    "RadiometricCalibration",
    "apply_calibration",
    "backproject_pixel",
    "calibrate_mosaic",
    "compute_quality_flags",
    "correct_bad_pixels",
    "crop_to_roi",
    "interleave_bands",
    "normalize_dn",
    "select_bands",
    "separate_bands",
]
