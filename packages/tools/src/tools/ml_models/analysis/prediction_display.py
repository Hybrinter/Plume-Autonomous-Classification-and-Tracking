"""Exact segmentation display arrays from immutable cached logits and explicit targets.

The raw-mask threshold is copied from frozen boundary provenance, not a
render setting. Stable float64 sigmoid display does not claim float32
flight probability rounding parity. Component IDs/matches remain frozen
spatial records; this module never matches or scores components again.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from flight.libs.types import Err, Ok, Result

from tools.ml_models.analysis.capture import CaptureRow
from tools.ml_models.analysis.metrics.inputs import sigmoid
from tools.ml_models.analysis.metrics.segmentation import _threshold
from tools.ml_models.analysis.model_figures import captured_values


@dataclass(frozen=True, slots=True)
class SegmentationDisplay:
    """Readonly H-by-W arrays for extent panels; matching comes from the supplied row."""

    truth: npt.NDArray[np.bool_]
    probability: npt.NDArray[np.float64]
    predicted: npt.NDArray[np.bool_]
    false_positive: npt.NDArray[np.bool_]
    false_negative: npt.NDArray[np.bool_]
    threshold: float


def segmentation_display_data(
    row: CaptureRow,
    target: npt.NDArray[np.float32],
    logits: npt.NDArray[np.float32],
) -> Result[SegmentationDisplay, str]:
    """Freeze exact display transforms and reject scalar/cache disagreement before rendering."""
    if row.key.task != "segmentor" or row.spatial is None:
        return Err(
            "segmentation visuals require recorded mask threshold and component/boundary evidence"
        )
    shape = (1, *row.key.spatial_shard)
    if (
        target.dtype != np.float32
        or logits.dtype != np.float32
        or target.shape != shape
        or logits.shape != shape
    ):
        return Err("segmentation preview arrays must be exact captured float32 (1,H,W)")
    if not np.isfinite(logits).all() or not np.isin(target, (0.0, 1.0)).all():
        return Err("segmentation previews need finite logits and an explicit binary target")
    threshold = row.spatial.boundary.mask_probability_threshold
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        return Err("segmentation preview has invalid frozen raw-mask threshold")
    score = logits.astype(np.float64).reshape(-1)
    truth = target[0].astype(np.bool_)
    predicted = _threshold(score, threshold).reshape(row.key.spatial_shard)
    probability = sigmoid(score).reshape(row.key.spatial_shard)
    fp, fn = predicted & ~truth, ~predicted & truth
    values = captured_values(row)
    expected = (
        ("target_area_px", np.count_nonzero(truth)),
        ("predicted_area_px", np.count_nonzero(predicted)),
        ("true_positive_pixels", np.count_nonzero(predicted & truth)),
        ("false_positive_pixels", np.count_nonzero(fp)),
        ("true_negative_pixels", np.count_nonzero(~predicted & ~truth)),
        ("false_negative_pixels", np.count_nonzero(fn)),
    )
    if any(values.get(name) != float(count) for name, count in expected):
        return Err("segmentation cached masks/logits disagree with frozen pixel evidence")
    for array in (truth, probability, predicted, fp, fn):
        array.setflags(write=False)
    return Ok(SegmentationDisplay(truth, probability, predicted, fp, fn, threshold))
