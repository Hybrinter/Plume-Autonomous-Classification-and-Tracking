"""Shared per-image localization/boundary measurement and frozen aggregate reduction.

Blob localization uses its own blob threshold/area gate. Boundary extent uses
the raw configured mask threshold without that area filter. This separation
keeps a tiny true component visible as a miss rather than erasing its truth.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from flight.libs.types import Err, Ok, Result

from tools.ml_models.analysis.config import ScoreConfig
from tools.ml_models.analysis.contracts import CurveEvidence, MetricValue
from tools.ml_models.analysis.metrics.boundary import (
    BoundaryRow,
    aggregate_boundary,
    score_boundary,
)
from tools.ml_models.analysis.metrics.inputs import binary_vectors
from tools.ml_models.analysis.metrics.localization import (
    LocalizationRow,
    aggregate_localization,
    score_localization,
)
from tools.ml_models.analysis.metrics.segmentation import _threshold


@dataclass(frozen=True, slots=True)
class SpatialRow:
    """Compact component matching and boundary evidence for one annotated image."""

    localization: LocalizationRow
    boundary: BoundaryRow


@dataclass(frozen=True, slots=True)
class SpatialEvidence:
    """Whole-cohort metrics and miss-inclusive exact success curves."""

    metrics: tuple[MetricValue, ...]
    curves: tuple[CurveEvidence, ...]


def score_spatial(
    logits: npt.ArrayLike,
    truth: npt.ArrayLike,
    *,
    gsd: tuple[float, float] | None = None,
    cfg: ScoreConfig | None = None,
) -> Result[SpatialRow, str]:
    """Measure components at blob settings and boundaries at raw-mask settings, without gating."""
    resolved = cfg if cfg is not None else ScoreConfig()
    local = score_localization(logits, truth, gsd=gsd, cfg=resolved)
    if isinstance(local, Err):
        return local
    try:
        score, target = np.asarray(logits), np.asarray(truth)
        validated = binary_vectors(score.reshape(-1), target.reshape(-1))
        if isinstance(validated, Err):
            return validated
        predicted = _threshold(validated.value[0], resolved.mask_probability_threshold).reshape(
            score.shape
        )
        boundary = score_boundary(predicted, target, gsd=gsd, cfg=resolved)
        if isinstance(boundary, Err):
            return boundary
        return Ok(SpatialRow(local.value, boundary.value))
    except (ValueError, TypeError, RuntimeError, OverflowError) as exc:
        return Err(f"spatial scoring failed: {exc}")


def aggregate_spatial(rows: tuple[SpatialRow, ...]) -> Result[SpatialEvidence, str]:
    """Reduce exact frozen sufficient statistics without inference or dense arrays."""
    local = aggregate_localization(tuple(row.localization for row in rows))
    if isinstance(local, Err):
        return local
    boundary = aggregate_boundary(tuple(row.boundary for row in rows))
    if isinstance(boundary, Err):
        return boundary
    return Ok(SpatialEvidence(local.value.metrics + boundary.value.metrics, local.value.curves))
