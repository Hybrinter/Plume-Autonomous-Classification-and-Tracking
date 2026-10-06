"""Unfiltered truth components, retained prediction components and cardinality-first matching.

Four-connected raster-order components are mask geometry, not independent
physical plumes. Eligible IoU matching maximizes match count before total
IoU. Conditional centroid errors are separate from miss-inclusive coverage.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from flight.libs.types import Err, Ok, Result
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from tools.ml_models.analysis.config import ScoreConfig
from tools.ml_models.analysis.contracts import CurveEvidence, MetricSupport, MetricValue, NamedCount
from tools.ml_models.analysis.metrics.boundary import validate_gsd
from tools.ml_models.analysis.metrics.inputs import binary_vectors
from tools.ml_models.analysis.metrics.segmentation import _threshold


@dataclass(frozen=True, slots=True)
class MaskComponent:
    """Raster-order four-connected mask component with unweighted pixel-center geometry."""

    component_id: int
    area_px: int
    area_m2: float | None
    centroid_x_px: float
    centroid_y_px: float
    bbox_xyxy: tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class ComponentMatch:
    """One eligible one-to-one pair; displacements are prediction minus truth."""

    truth_id: int
    prediction_id: int
    iou: float
    dx_px: float
    dy_px: float
    distance_px: float
    distance_m: float | None


@dataclass(frozen=True, slots=True)
class LocalizationRow:
    """Compact component/pair evidence; no dense predictions or boundaries are retained."""

    truth: tuple[MaskComponent, ...]
    predicted: tuple[MaskComponent, ...]
    matches: tuple[ComponentMatch, ...]
    unmatched_truth: tuple[int, ...]
    unmatched_prediction: tuple[int, ...]
    split_truth_components: int
    merge_predicted_components: int
    gsd_m: tuple[float, float] | None
    blob_probability_threshold: float
    min_blob_area_px: int
    match_iou_min: float
    method: str = "four_connected_raster_order_cardinality_then_total_iou"
    limitations: tuple[str, ...] = (
        "Truth components are not area-filtered and do not prove separate physical plumes.",
        "Prediction thresholding uses finite logits, not float32 flight probability rounding.",
        "Centroid distributions are conditional on eligible matching; "
        "success curves include misses.",
        "Ground distances and areas are local-GSD approximations.",
    )


@dataclass(frozen=True, slots=True)
class LocalizationEvidence:
    """Component counts, conditional errors and exact miss-inclusive success curves."""

    metrics: tuple[MetricValue, ...]
    curves: tuple[CurveEvidence, ...]


def match_components(
    ious: npt.ArrayLike,
    minimum: float,
) -> Result[tuple[tuple[int, int], ...], str]:
    """Maximize eligible match cardinality first, then summed IoU with zero-cost dummy matches."""
    try:
        raw = np.asarray(ious)
        if raw.ndim != 2 or raw.dtype.kind not in "iuf" or not np.isfinite(raw).all():
            return Err("component matching requires a finite two-dimensional numeric IoU matrix")
        if (
            np.any(raw < 0)
            or np.any(raw > 1)
            or isinstance(minimum, bool | np.bool_)
            or not math.isfinite(minimum)
            or not 0 <= minimum <= 1
        ):
            return Err("component IoUs and minimum must lie in [0,1]")
        n, m = raw.shape
        if min(n, m) == 0:
            return Ok(())
        cost = np.zeros((n + m, n + m), dtype=np.float64)
        eligible = raw >= minimum
        cost[:n, :m] = np.where(eligible, -(min(n, m) + 1 + raw), 1.0)
        truths, predictions = linear_sum_assignment(cost)
        return Ok(
            tuple(
                (int(a), int(b))
                for a, b in zip(truths, predictions, strict=True)
                if a < n and b < m and eligible[a, b]
            )
        )
    except (TypeError, ValueError, RuntimeError, OverflowError, MemoryError) as exc:
        return Err(f"component matching failed: {exc}")


def _components(
    foreground: npt.NDArray[np.bool_],
    minimum: int,
    gsd: tuple[float, float] | None,
) -> tuple[npt.NDArray[np.int32], tuple[MaskComponent, ...]]:
    """Label retained four-connected components, preserving original raster discovery order."""
    labelled, n = ndimage.label(foreground, structure=ndimage.generate_binary_structure(2, 1))
    areas = np.bincount(labelled.ravel(), minlength=n + 1)
    kept = tuple(index for index in range(1, n + 1) if areas[index] >= minimum)
    remap = np.zeros(n + 1, dtype=np.int32)
    remap[list(kept)] = np.arange(1, len(kept) + 1, dtype=np.int32)
    labels = remap[labelled]
    objects = ndimage.find_objects(labels)
    centers = ndimage.center_of_mass(foreground, labels, range(1, len(kept) + 1)) if kept else []
    records: list[MaskComponent] = []
    for index, (source_id, box, center) in enumerate(zip(kept, objects, centers, strict=True)):
        if box is None:
            raise ValueError("retained component has no bounding box")
        y, x = box
        area = int(areas[source_id])
        area_m2 = area * gsd[0] * gsd[1] if gsd is not None else None
        if area_m2 is not None and not math.isfinite(area_m2):
            raise ValueError("component local-GSD area exceeds finite range")
        records.append(
            MaskComponent(
                index,
                area,
                area_m2,
                float(center[1]),
                float(center[0]),
                (x.start, y.start, x.stop - 1, y.stop - 1),
            )
        )
    return labels, tuple(records)


def score_localization(
    logits: npt.ArrayLike,
    truth: npt.ArrayLike,
    *,
    gsd: tuple[float, float] | None = None,
    cfg: ScoreConfig | None = None,
) -> Result[LocalizationRow, str]:
    """Measure every truth component and retained predicted blob without classifier gating."""
    geometry = validate_gsd(gsd)
    if isinstance(geometry, Err):
        return geometry
    resolved = cfg if cfg is not None else ScoreConfig()
    try:
        raw_score, raw_truth = np.asarray(logits), np.asarray(truth)
        if raw_score.shape != raw_truth.shape:
            return Err("localization logits and truth require aligned shapes")
        if raw_score.ndim == 3 and raw_score.shape[0] == 1:
            raw_score, raw_truth = raw_score[0], raw_truth[0]
        if raw_score.ndim != 2 or min(raw_score.shape) < 1:
            return Err("localization requires one nonempty H-by-W mask")
        validated = binary_vectors(raw_score.reshape(-1), raw_truth.reshape(-1))
        if isinstance(validated, Err):
            return validated
        score, target = validated.value
        predicted = _threshold(score, resolved.blob_probability_threshold).reshape(raw_score.shape)
        labels_a, components_a = _components(target.reshape(raw_score.shape), 1, geometry.value)
        labels_b, components_b = _components(predicted, resolved.min_blob_area_px, geometry.value)
        n, m = len(components_a), len(components_b)
        intersections = np.bincount(
            (labels_a.astype(np.int64) * (m + 1) + labels_b).ravel(),
            minlength=(n + 1) * (m + 1),
        ).reshape(n + 1, m + 1)[1:, 1:]
        areas_a = np.array([c.area_px for c in components_a], dtype=np.int64)
        areas_b = np.array([c.area_px for c in components_b], dtype=np.int64)
        union = areas_a[:, None] + areas_b[None, :] - intersections
        ious = intersections / union if n and m else np.zeros((n, m), dtype=np.float64)
        assigned = match_components(ious, resolved.match_iou_min)
        if isinstance(assigned, Err):
            return assigned
        matches: list[ComponentMatch] = []
        for a, b in assigned.value:
            dx = components_b[b].centroid_x_px - components_a[a].centroid_x_px
            dy = components_b[b].centroid_y_px - components_a[a].centroid_y_px
            distance_m = (
                math.hypot(dx * geometry.value[0], dy * geometry.value[1])
                if geometry.value is not None
                else None
            )
            if distance_m is not None and not math.isfinite(distance_m):
                return Err("matched centroid ground distance exceeds finite range")
            matches.append(
                ComponentMatch(a, b, float(ious[a, b]), dx, dy, math.hypot(dx, dy), distance_m)
            )
        return Ok(
            LocalizationRow(
                components_a,
                components_b,
                tuple(matches),
                tuple(index for index in range(n) if index not in {a for a, _ in assigned.value}),
                tuple(index for index in range(m) if index not in {b for _, b in assigned.value}),
                int(np.count_nonzero((intersections > 0).sum(axis=1) > 1)),
                int(np.count_nonzero((intersections > 0).sum(axis=0) > 1)),
                geometry.value,
                resolved.blob_probability_threshold,
                resolved.min_blob_area_px,
                resolved.match_iou_min,
            )
        )
    except (TypeError, ValueError, RuntimeError, OverflowError, MemoryError) as exc:
        return Err(f"localization scoring failed: {exc}")


def _metric(
    name: str,
    value: float | int | None,
    support: MetricSupport,
    *,
    unit: str = "dimensionless",
    aggregation: str,
    reason: str = "no eligible component support",
) -> MetricValue:
    """Keep undefined rates and conditional distances explicitly unavailable."""
    return MetricValue(
        name=name,
        value=value,
        status="AVAILABLE" if value is not None else "UNAVAILABLE",
        reason=None if value is not None else reason,
        support=support,
        unit=unit,
        aggregation=aggregation,
    )


def aggregate_localization(rows: tuple[LocalizationRow, ...]) -> Result[LocalizationEvidence, str]:
    """Pool component counts and retain all truth misses in exact localization success curves."""
    if not rows:
        return Err("localization aggregation requires annotated images")
    for row in rows:
        geometry = validate_gsd(row.gsd_m)
        if isinstance(geometry, Err):
            return geometry
        n_truth, n_prediction = len(row.truth), len(row.predicted)
        truth_ids, prediction_ids = (
            {m.truth_id for m in row.matches},
            {m.prediction_id for m in row.matches},
        )
        if (
            len(truth_ids) != len(row.matches)
            or len(prediction_ids) != len(row.matches)
            or not truth_ids <= set(range(n_truth))
            or not prediction_ids <= set(range(n_prediction))
            or tuple(row.unmatched_truth) != tuple(i for i in range(n_truth) if i not in truth_ids)
            or tuple(row.unmatched_prediction)
            != tuple(i for i in range(n_prediction) if i not in prediction_ids)
            or not 0 <= row.split_truth_components <= n_truth
            or not 0 <= row.merge_predicted_components <= n_prediction
        ):
            return Err("invalid frozen component assignment or unmatched counts")
        for components in (row.truth, row.predicted):
            if any(
                c.component_id != index
                or type(c.area_px) is not int
                or c.area_px < 1
                or not math.isfinite(c.centroid_x_px)
                or not math.isfinite(c.centroid_y_px)
                or c.area_m2 is not None
                and (not math.isfinite(c.area_m2) or c.area_m2 <= 0)
                for index, c in enumerate(components)
            ):
                return Err("invalid frozen component geometry")
        if any(
            not math.isfinite(m.iou)
            or not row.match_iou_min <= m.iou <= 1
            or not all(math.isfinite(v) for v in (m.dx_px, m.dy_px, m.distance_px))
            or m.distance_px < 0
            or m.distance_m is not None
            and (not math.isfinite(m.distance_m) or m.distance_m < 0)
            or (m.distance_m is None) != (row.gsd_m is None)
            for m in row.matches
        ):
            return Err("invalid frozen matched-component distances or IoU")
    first = rows[0]
    if any(
        (r.blob_probability_threshold, r.min_blob_area_px, r.match_iou_min)
        != (first.blob_probability_threshold, first.min_blob_area_px, first.match_iou_min)
        for r in rows
    ):
        return Err("localization rows disagree on component/matching conventions")
    truths, predictions = sum(len(r.truth) for r in rows), sum(len(r.predicted) for r in rows)
    matches = tuple(match for row in rows for match in row.matches)
    n_matched = len(matches)
    truth_support = MetricSupport(
        unit="COMPONENT", n=truths, counts=(NamedCount(name="matched", value=n_matched),)
    )
    prediction_support = MetricSupport(
        unit="COMPONENT", n=predictions, counts=(NamedCount(name="matched", value=n_matched),)
    )
    image_support = MetricSupport(unit="IMAGE", n=len(rows))
    metrics = tuple(
        _metric(name, value, support, aggregation=aggregation, unit=unit)
        for name, value, support, aggregation, unit in (
            ("truth_components", truths, image_support, "exact_component_count", "component"),
            (
                "predicted_components",
                predictions,
                image_support,
                "exact_component_count",
                "component",
            ),
            ("matched_components", n_matched, truth_support, "exact_component_count", "component"),
            (
                "unmatched_truth_components",
                truths - n_matched,
                truth_support,
                "exact_component_count",
                "component",
            ),
            (
                "unmatched_prediction_components",
                predictions - n_matched,
                prediction_support,
                "exact_component_count",
                "component",
            ),
            (
                "component_precision",
                n_matched / predictions if predictions else None,
                prediction_support,
                "pooled_retained_prediction_match_fraction",
                "dimensionless",
            ),
            (
                "component_recall",
                n_matched / truths if truths else None,
                truth_support,
                "pooled_truth_match_fraction",
                "dimensionless",
            ),
            (
                "component_f1",
                2 * n_matched / (truths + predictions) if truths + predictions else None,
                image_support,
                "pooled_component_f1",
                "dimensionless",
            ),
            (
                "split_truth_components",
                sum(r.split_truth_components for r in rows),
                truth_support,
                "truths_overlapping_multiple_retained_predictions",
                "component",
            ),
            (
                "merge_predicted_components",
                sum(r.merge_predicted_components for r in rows),
                prediction_support,
                "retained_predictions_overlapping_multiple_truths",
                "component",
            ),
        )
    )
    curves: list[CurveEvidence] = []
    for unit in ("px", "m"):
        values = tuple(
            value
            for match in matches
            if (value := match.distance_px if unit == "px" else match.distance_m) is not None
        )
        measured_truths = (
            truths if unit == "px" else sum(len(r.truth) for r in rows if r.gsd_m is not None)
        )
        matched_support = MetricSupport(unit="COMPONENT", n=len(values))
        for reduction, value in (
            ("mean", math.fsum(value / len(values) for value in values) if values else None),
            ("median", float(np.median(values)) if values else None),
            ("p95", float(np.quantile(values, 0.95, method="linear")) if values else None),
        ):
            metrics += (
                _metric(
                    "matched_centroid_error_" + unit + "_" + reduction,
                    value,
                    matched_support,
                    unit="pixel" if unit == "px" else "m",
                    aggregation="matched_component_conditional_" + reduction,
                    reason="no eligible matches with requested distance geometry",
                ),
            )
        if measured_truths:
            thresholds = tuple(sorted({0.0, *values}))
            curves.append(
                CurveEvidence(
                    name="localization_success_" + unit,
                    x_name="centroid_error_tolerance",
                    y_name="truth_component_success_fraction",
                    x=thresholds,
                    y=tuple(
                        sum(value <= tolerance for value in values) / measured_truths
                        for tolerance in thresholds
                    ),
                    x_unit="pixel" if unit == "px" else "m",
                    y_unit="fraction",
                    support=MetricSupport(
                        unit="COMPONENT",
                        n=measured_truths,
                        counts=(
                            NamedCount(name="total_truth_components", value=truths),
                            NamedCount(
                                name="truth_components_missing_geometry",
                                value=truths - measured_truths,
                            ),
                        ),
                    ),
                    notes=(
                        "Denominator includes every truth component with the requested geometry, "
                        "including unmatched misses.",
                        "Metre coverage excludes truth components without recorded GSD and reports "
                        "that missing support explicitly.",
                        "Coordinates use observed matched distances; "
                        "they are not mission acceptance tolerances.",
                    ),
                )
            )
    return Ok(LocalizationEvidence(metrics, tuple(curves)))
