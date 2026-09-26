"""Payload preprocessing: pure functions transforming a prism buffer for inference.

Stage order in the payload loop: stack_channels -> calibrate_mosaic -> normalize_dn ->
select_bands -> compute_quality_flags. All functions are pure (no I/O, no global
state). Calibration runs on the stacked (3, H, W) planes. The published tensor is
NCHW (1, C, H, W) at the full sensor size, with no crop and no scale.

Satisfies: REQ-AIML-PREP-001, REQ-AIML-PREP-002, REQ-AIML-IMAG-001.
"""

from flight.payload.preprocess.band_select import select_bands
from flight.payload.preprocess.normalize import normalize_dn
from flight.payload.preprocess.quality import SmearRateSource, compute_quality_flags
from flight.payload.preprocess.radiometric import (
    MosaicCalibration,
    calibrate_mosaic,
    correct_bad_pixels,
)
from flight.payload.preprocess.stack import stack_channels

__all__ = [
    "MosaicCalibration",
    "SmearRateSource",
    "calibrate_mosaic",
    "compute_quality_flags",
    "correct_bad_pixels",
    "normalize_dn",
    "select_bands",
    "stack_channels",
]
