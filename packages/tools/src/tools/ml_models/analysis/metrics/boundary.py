"""Explicit interior mask boundaries, inclusive tolerance hits and symmetric distances.

Four-neighbour erosion treats outside-image pixels as background. Distances
pool both directed nearest-boundary pixel distances; HD95 is their linear
95th percentile. Empty-boundary distances are unavailable, not fabricated.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from flight.libs.types import Err, Ok, Result
from scipy import ndimage

from tools.ml_models.analysis.config import ScoreConfig
from tools.ml_models.analysis.contracts import MetricSupport, MetricValue, NamedCount


@dataclass(frozen=True, slots=True)
class BoundaryRow:
    """One explicit mask pair; hits use the stated tolerance, distances need both boundaries."""

    truth_boundary_pixels: int
    predicted_boundary_pixels: int
    truth_hits: int
    predicted_hits: int
    tolerance: float
    tolerance_unit: str
    asd_px: float | None
    hd95_px: float | None
    asd_m: float | None
    hd95_m: float | None
    distance_reason: str | None
    ground_distance_reason: str | None
    method: str = "interior_four_neighbor_boundary_pooled_directed_distances_linear_hd95"
    mask_probability_threshold: float = 0.5


@dataclass(frozen=True, slots=True)
class BoundaryEvidence:
    """Pooled boundary hits plus equal-image conditional distance summaries."""

    metrics: tuple[MetricValue, ...]


def binary_masks(
    predicted: npt.ArrayLike,
    truth: npt.ArrayLike,
) -> Result[tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]], str]:
    """Validate one aligned binary H-by-W or singleton-channel explicit mask pair."""
    try:
        raw_prediction, raw_truth = np.asarray(predicted), np.asarray(truth)
        if raw_prediction.shape != raw_truth.shape:
            return Err("boundary masks require aligned shapes")
        if raw_prediction.ndim == 3 and raw_prediction.shape[0] == 1:
            raw_prediction, raw_truth = raw_prediction[0], raw_truth[0]
        if raw_prediction.ndim != 2 or min(raw_prediction.shape) < 1:
            return Err("boundary masks require nonempty H-by-W binary arrays")
        if any(
            value.dtype.kind not in "biuf" or not np.isin(value, (0, 1)).all()
            for value in (raw_prediction, raw_truth)
        ):
            return Err("boundary masks must be exactly binary")
        return Ok((raw_prediction.astype(np.bool_), raw_truth.astype(np.bool_)))
    except (TypeError, ValueError, OverflowError) as exc:
        return Err(f"invalid boundary masks: {exc}")


def validate_gsd(gsd: tuple[float, float] | None) -> Result[tuple[float, float] | None, str]:
    """Keep unknown ground geometry unavailable; reject supplied invalid axes or area."""
    if gsd is None:
        return Ok(None)
    try:
        if (
            len(gsd) != 2
            or any(
                isinstance(value, bool | np.bool_) or not math.isfinite(value) or value <= 0
                for value in gsd
            )
            or not math.isfinite(gsd[0] * gsd[1])
        ):
            return Err("spatial evidence requires finite positive lateral/along GSD")
        return Ok((float(gsd[0]), float(gsd[1])))
    except (TypeError, ValueError, OverflowError) as exc:
        return Err(f"invalid spatial GSD: {exc}")


def score_boundary(
    predicted: npt.ArrayLike,
    truth: npt.ArrayLike,
    *,
    gsd: tuple[float, float] | None = None,
    cfg: ScoreConfig | None = None,
) -> Result[BoundaryRow, str]:
    """Measure exact boundary hits and pixel/local-ground distances for explicit binary masks."""
    validated = binary_masks(predicted, truth)
    geometry = validate_gsd(gsd)
    if isinstance(validated, Err):
        return validated
    if isinstance(geometry, Err):
        return geometry
    resolved = cfg if cfg is not None else ScoreConfig()
    axes = geometry.value
    if resolved.boundary_tolerance_m is not None and axes is None:
        return Err("physical boundary tolerance requires recorded GSD")
    try:
        prediction, target = validated.value
        structure = ndimage.generate_binary_structure(2, 1)
        boundary_a = target & ~ndimage.binary_erosion(target, structure=structure, border_value=0)
        boundary_b = prediction & ~ndimage.binary_erosion(
            prediction, structure=structure, border_value=0
        )
        n_a, n_b = int(boundary_a.sum()), int(boundary_b.sum())
        tolerance = (
            resolved.boundary_tolerance_m
            if resolved.boundary_tolerance_m is not None
            else resolved.boundary_tolerance_px
        )
        unit = "m" if resolved.boundary_tolerance_m is not None else "pixel"
        if n_a == 0 or n_b == 0:
            reason = "one or both explicit masks have no boundary pixels"
            return Ok(
                BoundaryRow(
                    n_a,
                    n_b,
                    0,
                    0,
                    tolerance,
                    unit,
                    None,
                    None,
                    None,
                    None,
                    reason,
                    reason,
                    mask_probability_threshold=resolved.mask_probability_threshold,
                )
            )
        distances_a = ndimage.distance_transform_edt(~boundary_b)[boundary_a]
        distances_b = ndimage.distance_transform_edt(~boundary_a)[boundary_b]
        pooled = np.concatenate((distances_a, distances_b))
        ground_a = ground_b = None
        asd_m = hd95_m = None
        if axes is not None:
            sampling = (axes[1], axes[0])
            ground_a = ndimage.distance_transform_edt(~boundary_b, sampling=sampling)[boundary_a]
            ground_b = ndimage.distance_transform_edt(~boundary_a, sampling=sampling)[boundary_b]
            ground = np.concatenate((ground_a, ground_b))
            asd_m = math.fsum(float(value) / len(ground) for value in ground)
            hd95_m = float(np.quantile(ground, 0.95, method="linear"))
            if not math.isfinite(asd_m) or not math.isfinite(hd95_m):
                return Err("boundary ground distances exceed finite range")
        hits_a = ground_a if unit == "m" else distances_a
        hits_b = ground_b if unit == "m" else distances_b
        if hits_a is None or hits_b is None:
            return Err("physical boundary distances unavailable")
        return Ok(
            BoundaryRow(
                n_a,
                n_b,
                int(np.count_nonzero(hits_a <= tolerance)),
                int(np.count_nonzero(hits_b <= tolerance)),
                tolerance,
                unit,
                math.fsum(float(value) / len(pooled) for value in pooled),
                float(np.quantile(pooled, 0.95, method="linear")),
                asd_m,
                hd95_m,
                None,
                None if axes is not None else "recorded GSD unavailable",
                mask_probability_threshold=resolved.mask_probability_threshold,
            )
        )
    except (TypeError, ValueError, RuntimeError, OverflowError, FloatingPointError) as exc:
        return Err(f"boundary scoring failed: {exc}")


def _metric(
    name: str,
    value: float | int | None,
    support: MetricSupport,
    *,
    unit: str = "dimensionless",
    aggregation: str,
    reason: str = "no eligible boundary support",
) -> MetricValue:
    """Emit finite measured values or explicit missing-support reasons."""
    return MetricValue(
        name=name,
        value=value,
        status="AVAILABLE" if value is not None else "UNAVAILABLE",
        reason=None if value is not None else reason,
        support=support,
        unit=unit,
        aggregation=aggregation,
    )


def aggregate_boundary(rows: tuple[BoundaryRow, ...]) -> Result[BoundaryEvidence, str]:
    """Aggregate pooled tolerance hits and conditional equal-image ASD/HD95 without empty
    inflation."""
    if not rows:
        return Err("boundary aggregation requires annotated images")
    for row in rows:
        counts = (
            row.truth_boundary_pixels,
            row.predicted_boundary_pixels,
            row.truth_hits,
            row.predicted_hits,
        )
        distances = (row.asd_px, row.hd95_px, row.asd_m, row.hd95_m)
        both = row.truth_boundary_pixels > 0 and row.predicted_boundary_pixels > 0
        if (
            any(type(count) is not int or count < 0 for count in counts)
            or row.truth_hits > row.truth_boundary_pixels
            or row.predicted_hits > row.predicted_boundary_pixels
            or any(
                value is not None
                and (isinstance(value, bool) or not math.isfinite(value) or value < 0)
                for value in distances
            )
            or (row.asd_px is not None) != both
            or (row.hd95_px is not None) != both
            or (row.asd_m is None) != (row.hd95_m is None)
            or not math.isfinite(row.tolerance)
            or row.tolerance <= 0
            or row.tolerance_unit not in ("pixel", "m")
            or not math.isfinite(row.mask_probability_threshold)
            or not 0 <= row.mask_probability_threshold <= 1
        ):
            return Err("invalid frozen boundary counts, distances or tolerance")
    first = rows[0]
    if any(
        (r.tolerance, r.tolerance_unit, r.mask_probability_threshold)
        != (first.tolerance, first.tolerance_unit, first.mask_probability_threshold)
        for r in rows
    ):
        return Err("boundary rows disagree on tolerance convention")
    n = len(rows)
    truth, prediction = (
        sum(r.truth_boundary_pixels for r in rows),
        sum(r.predicted_boundary_pixels for r in rows),
    )
    hits_truth, hits_prediction = (
        sum(r.truth_hits for r in rows),
        sum(r.predicted_hits for r in rows),
    )
    precision = hits_prediction / prediction if prediction else None
    recall = hits_truth / truth if truth else None
    harmonic = (
        2 * hits_truth * hits_prediction / (hits_prediction * truth + hits_truth * prediction)
        if hits_prediction * truth + hits_truth * prediction
        else 0.0
        if truth + prediction
        else None
    )
    image_support = MetricSupport(unit="IMAGE", n=n)
    truth_support = MetricSupport(
        unit="PIXEL", n=truth, counts=(NamedCount(name="n_images", value=n),)
    )
    prediction_support = MetricSupport(
        unit="PIXEL", n=prediction, counts=(NamedCount(name="n_images", value=n),)
    )
    metrics: tuple[MetricValue, ...] = (
        _metric(
            "boundary_precision",
            precision,
            prediction_support,
            aggregation="pooled_predicted_boundary_hit_fraction",
        ),
        _metric(
            "boundary_recall",
            recall,
            truth_support,
            aggregation="pooled_truth_boundary_hit_fraction",
        ),
        _metric(
            "boundary_f1", harmonic, image_support, aggregation="harmonic_pooled_boundary_hit_rates"
        ),
        _metric(
            "boundary_missed_images",
            sum(r.truth_boundary_pixels > 0 and r.predicted_boundary_pixels == 0 for r in rows),
            image_support,
            unit="image",
            aggregation="exact_image_count",
        ),
        _metric(
            "boundary_spurious_images",
            sum(r.truth_boundary_pixels == 0 and r.predicted_boundary_pixels > 0 for r in rows),
            image_support,
            unit="image",
            aggregation="exact_image_count",
        ),
        _metric(
            "boundary_empty_images",
            sum(r.truth_boundary_pixels == r.predicted_boundary_pixels == 0 for r in rows),
            image_support,
            unit="image",
            aggregation="exact_image_count",
        ),
    )
    for name, candidates in (
        ("asd_px", tuple(r.asd_px for r in rows)),
        ("hd95_px", tuple(r.hd95_px for r in rows)),
        ("asd_m", tuple(r.asd_m for r in rows)),
        ("hd95_m", tuple(r.hd95_m for r in rows)),
    ):
        values = tuple(value for value in candidates if value is not None)
        metrics += (
            _metric(
                "boundary_" + name + "_mean",
                math.fsum(value / len(values) for value in values) if values else None,
                MetricSupport(unit="IMAGE", n=len(values)),
                unit="m" if name.endswith("_m") else "pixel",
                aggregation="equal_image_mean_conditional_on_both_nonempty_boundaries",
                reason="both nonempty boundaries and the requested distance geometry are required",
            ),
        )
    return Ok(BoundaryEvidence(metrics))
