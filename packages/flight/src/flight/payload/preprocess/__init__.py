"""Payload preprocessing: pure functions from a raw RGB stack to a model tensor.

Stage order: calibrate_mosaic -> confirm_planes -> normalize_dn ->
compute_quality_flags -> upsample_planes. Channel order is ``BAND_ORDER``.
There is no per-frame reorder and no mosaic unpack.

Satisfies: REQ-AIML-PREP-001, REQ-AIML-PREP-002, REQ-AIML-IMAG-001.
"""

from flight.payload.preprocess.band_select import canonical_band_names
from flight.payload.preprocess.demosaic import confirm_planes
from flight.payload.preprocess.normalize import normalize_dn
from flight.payload.preprocess.quality import SmearRateSource, compute_quality_flags
from flight.payload.preprocess.radiometric import (
    MosaicCalibration,
    calibrate_mosaic,
    correct_bad_pixels,
    dark_matches_frame,
)
from flight.payload.preprocess.resample import upsample_planes

__all__ = [
    "MosaicCalibration",
    "SmearRateSource",
    "calibrate_mosaic",
    "canonical_band_names",
    "compute_quality_flags",
    "confirm_planes",
    "correct_bad_pixels",
    "dark_matches_frame",
    "normalize_dn",
    "upsample_planes",
]
