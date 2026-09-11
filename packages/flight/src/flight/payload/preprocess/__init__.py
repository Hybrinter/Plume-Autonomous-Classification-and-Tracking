"""Payload preprocessing: pure functions transforming a raw mosaic plane for inference.

Stage order in the payload loop: calibrate_mosaic -> separate_bands -> normalize_dn ->
select_bands -> compute_quality_flags. The full demosaiced band plane is passed to
inference with no crop and no scale.

All functions are pure (no I/O, no global state). Calibration and raw-DN saturation run
on the raw mosaic plane (pre-demosaic) where the sensor physics lives; quality flags run
on the full band plane before any ROI crop or decimation. Beyond the pipeline stages the
package provides the keep/drop gate (decide_usability mapping compute_quality_flags to
TRAINING | TRACKING | INVALID under a UsabilityPolicy) and ROI resampling helpers
(decimate_to_size / crop_and_upsample with a RoiTransform record and plane<->tensor
pixel backprojection) for search/track frame preparation.

Satisfies: REQ-AIML-PREP-001, REQ-AIML-PREP-002, REQ-AIML-PREP-003, REQ-AIML-IMAG-001,
REQ-AIML-IMAG-002, REQ-AIML-DATA-003, REQ-AIML-DATA-005.
"""

from flight.payload.preprocess.band_select import band_index, select_bands
from flight.payload.preprocess.crop import (
    RoiTransform,
    crop_and_upsample,
    crop_plane,
    decimate_area,
    decimate_to_size,
    plane_to_tensor_px,
    tensor_to_plane_px,
    upsample,
)
from flight.payload.preprocess.demosaic import CELL_OFFSETS, interleave_bands, separate_bands
from flight.payload.preprocess.normalize import (
    full_scale_dn,
    normalize_dn,
    scale_to_reference_exposure,
)
from flight.payload.preprocess.quality import (
    DEFAULT_BAND_NAMES,
    DEFAULT_USABILITY_POLICY,
    SATURATION_PIXEL_LEVEL,
    CloudTest,
    QualityMetrics,
    UsabilityPolicy,
    cloud_fraction,
    compute_quality_flags,
    compute_quality_metrics,
    decide_usability,
    flags_from_metrics,
    predicted_smear_px,
    saturated_fraction_normalized,
    saturated_fraction_raw,
)
from flight.payload.preprocess.radiometric import (
    MosaicCalibration,
    calibrate_mosaic,
    correct_bad_pixels,
    dark_matches_frame,
)

__all__ = [
    "CELL_OFFSETS",
    "DEFAULT_BAND_NAMES",
    "DEFAULT_USABILITY_POLICY",
    "SATURATION_PIXEL_LEVEL",
    "CloudTest",
    "MosaicCalibration",
    "QualityMetrics",
    "RoiTransform",
    "UsabilityPolicy",
    "band_index",
    "calibrate_mosaic",
    "cloud_fraction",
    "compute_quality_flags",
    "compute_quality_metrics",
    "correct_bad_pixels",
    "crop_and_upsample",
    "crop_plane",
    "dark_matches_frame",
    "decide_usability",
    "decimate_area",
    "decimate_to_size",
    "flags_from_metrics",
    "full_scale_dn",
    "interleave_bands",
    "normalize_dn",
    "plane_to_tensor_px",
    "predicted_smear_px",
    "saturated_fraction_normalized",
    "saturated_fraction_raw",
    "scale_to_reference_exposure",
    "select_bands",
    "separate_bands",
    "tensor_to_plane_px",
    "upsample",
]
