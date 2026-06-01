"""Payload preprocessing: pure functions transforming raw bands for inference.

Stage order in the payload loop: select bands -> radiometric correction ->
quality flags -> crop. All functions are pure (no I/O, no global state).
"""

from flight.payload.preprocess.band_select import BAND_INDICES, select_bands
from flight.payload.preprocess.crop import backproject_pixel, crop_to_roi
from flight.payload.preprocess.quality import compute_quality_flags
from flight.payload.preprocess.radiometric import RadiometricCalibration, apply_calibration

__all__ = [
    "BAND_INDICES",
    "RadiometricCalibration",
    "apply_calibration",
    "backproject_pixel",
    "compute_quality_flags",
    "crop_to_roi",
    "select_bands",
]
