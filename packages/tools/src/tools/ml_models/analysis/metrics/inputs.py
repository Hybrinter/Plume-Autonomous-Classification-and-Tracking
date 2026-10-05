"""Validated binary vectors and stable numeric transforms for pure scoring.

Inputs are aligned vectors or N-by-one columns. Targets are exactly binary;
scores are finite numeric values, not strings or booleans.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from flight.libs.types import Err, Ok, Result

type FloatVector = npt.NDArray[np.float64]
type BoolVector = npt.NDArray[np.bool_]


def binary_vectors(
    scores: npt.ArrayLike,
    labels: npt.ArrayLike,
    *,
    probabilities: bool = False,
) -> Result[tuple[FloatVector, BoolVector], str]:
    """Validate aligned finite scores and exact binary labels without clipping."""
    try:
        raw_scores, raw_labels = np.asarray(scores), np.asarray(labels)
        for array in (raw_scores, raw_labels):
            if array.ndim not in (1, 2) or (array.ndim == 2 and array.shape[1] != 1):
                return Err("binary scoring requires vectors or N-by-one columns")
        if raw_scores.dtype.kind not in "iuf" or raw_labels.dtype.kind not in "biuf":
            return Err("scores must be numeric and labels must be binary")
        if raw_scores.dtype.kind in "iu" and np.any(np.abs(raw_scores.astype(object)) > 2**53):
            return Err("integer scores exceed exact float64 representation")
        score = raw_scores.astype(np.float64, copy=False).reshape(-1)
        target = raw_labels.astype(np.float64, copy=False).reshape(-1)
    except (TypeError, ValueError, OverflowError) as exc:
        return Err(f"invalid binary vectors: {exc}")
    if score.size == 0 or score.size != target.size:
        return Err("binary scoring requires nonempty aligned inputs")
    if not np.isfinite(score).all() or not np.isfinite(target).all():
        return Err("binary scoring requires finite inputs")
    if not np.isin(target, (0.0, 1.0)).all():
        return Err("targets must be exactly binary")
    if probabilities and np.any((score < 0.0) | (score > 1.0)):
        return Err("probabilities must lie in [0, 1]")
    return Ok((score, target.astype(np.bool_)))


def sigmoid(logits: FloatVector) -> FloatVector:
    """Compute stable probabilities without changing the raw ranking scores."""
    positive = logits >= 0.0
    probabilities = np.empty_like(logits)
    probabilities[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exponential = np.exp(logits[~positive])
    probabilities[~positive] = exponential / (1.0 + exponential)
    return probabilities


def finite_mean(values: FloatVector) -> float:
    """Average a nonempty finite vector without overflowing a preliminary sum."""
    return math.fsum(float(value) / len(values) for value in values)
