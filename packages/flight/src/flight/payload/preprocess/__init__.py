"""Payload preprocessing: pure functions transforming a raw mosaic plane for inference.

Stage order in the payload loop: calibrate_mosaic -> separate_bands -> normalize_dn ->
select_bands -> compute_quality_flags -> crop. All functions are pure (no I/O, no global
state); calibration runs on the raw mosaic plane (pre-demosaic) where the physics lives.

Satisfies: REQ-AIML-PREP-001, REQ-AIML-PREP-002, REQ-AIML-IMAG-001.
"""

from flight.payload.preprocess.band_select import select_bands
from flight.payload.preprocess.crop import backproject_pixel, crop_to_roi
from flight.payload.preprocess.demosaic import CELL_OFFSETS, interleave_bands, separate_bands
from flight.payload.preprocess.normalize import normalize_dn
from flight.payload.preprocess.quality import compute_quality_flags
from flight.payload.preprocess.radiometric import (
    MosaicCalibration,
    calibrate_mosaic,
    correct_bad_pixels,
)

__all__ = [
    "CELL_OFFSETS",
    "MosaicCalibration",
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
