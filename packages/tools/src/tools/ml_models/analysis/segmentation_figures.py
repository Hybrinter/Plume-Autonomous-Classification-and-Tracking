"""Extent and localization chart recipes from frozen explicit-mask evidence.

Positive-mask distributions stay separate from verified negatives and
unverified empty annotations. Matched centroid ECDFs are conditional;
copied localization-success curves retain misses in their denominator.
No component matching, boundary measurement or dense inference is rerun.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import math

from flight.libs.types import Err, Ok, Result

from tools.ml_models.analysis.capture import CaptureRow
from tools.ml_models.analysis.contracts import SplitEvidence
from tools.ml_models.analysis.model_figures import (
    DrawStyle,
    ModelFigure,
    ModelPoint,
    captured_values,
    curve_figure,
    distribution_figure,
    figure_identity,
    metric_figure,
    validate_figure_rows,
)


def segmentation_figure_data(
    evidence: SplitEvidence,
    rows: tuple[CaptureRow, ...],
) -> Result[tuple[ModelFigure, ...], str]:
    """Freeze all segmentation curve/scalar families with exact populations and units."""
    if evidence.task != "segmentor":
        return Err("segmentation figures require segmentor evidence")
    checked = validate_figure_rows(evidence, rows)
    if isinstance(checked, Err):
        return checked
    records = tuple(captured_values(row) for row in rows)
    required = (
        "target_area_px",
        "predicted_area_px",
        "true_positive_pixels",
        "false_positive_pixels",
        "true_negative_pixels",
        "false_negative_pixels",
        "foreground_iou",
        "foreground_dice",
    )
    if any(any(record.get(name) is None for name in required) for record in records):
        return Err("segmentation figures need complete explicit-mask scalar capture")
    for row, record in zip(rows, records, strict=True):
        pixels = row.key.spatial_shard[0] * row.key.spatial_shard[1]
        tp, fp, tn, fn = tuple(record[name] for name in required[2:6])
        if any(value is None or value < 0 or not value.is_integer() for value in (tp, fp, tn, fn)):
            return Err("segmentation captured pixel counts must be nonnegative integers")
        if tp is None or fp is None or tn is None or fn is None:
            return Err("segmentation captured pixel counts are missing")
        if (
            tp + fp + tn + fn != pixels
            or record["target_area_px"] != tp + fn
            or record["predicted_area_px"] != tp + fp
        ):
            return Err("segmentation captured areas disagree with explicit-mask pixel counts")
        union, dice_denominator = tp + fp + fn, 2 * tp + fp + fn
        for name, expected in (
            ("foreground_iou", tp / union if union else 1.0),
            ("foreground_dice", 2 * tp / dice_denominator if dice_denominator else 1.0),
        ):
            recorded = record[name]
            if recorded is None or abs(recorded - expected) > 1e-12:
                return Err(
                    "segmentation captured overlap disagrees with explicit-mask pixel counts"
                )
    positive = tuple(
        index
        for index, record in enumerate(records)
        if record["target_area_px"] is not None and record["target_area_px"] > 0
    )
    negative = tuple(
        index
        for index, (row, record) in enumerate(zip(rows, records, strict=True))
        if row.label == 0 and record["target_area_px"] == 0
    )
    other_empty = tuple(
        index
        for index, (row, record) in enumerate(zip(rows, records, strict=True))
        if row.label == 1 and record["target_area_px"] == 0
    )
    cohorts = (
        ("positive truth masks", positive),
        ("verified negative empty masks", negative),
        ("positive-label empty masks", other_empty),
    )
    figures: list[ModelFigure] = []
    for name, label, unit, positive_only in (
        ("foreground_iou", "Foreground IoU", "fraction", True),
        ("foreground_dice", "Foreground Dice", "fraction", True),
        ("binary_cross_entropy", "Unweighted image BCE", "dimensionless", False),
        ("brier_score", "Unweighted image Brier", "dimensionless", False),
        ("objective_loss", "Configured weighted image objective", "dimensionless", False),
        ("target_area_px", "Explicit truth foreground area", "pixel", False),
        ("predicted_area_px", "Raw-mask predicted foreground area", "pixel", False),
        ("target_area_m2", "Local-GSD truth foreground area", "m2", False),
        ("predicted_area_m2", "Local-GSD predicted foreground area", "m2", False),
        ("predicted_blobs", "Retained predicted blob count", "component/image", False),
    ):
        groups = cohorts[:1] if positive_only else cohorts
        populations = tuple(
            (
                label_name,
                tuple(
                    value for index in indices if (value := records[index].get(name)) is not None
                ),
            )
            for label_name, indices in groups
        )
        figures.append(
            distribution_figure(
                evidence,
                name + "_distribution",
                "Captured " + label,
                label + " (" + unit + ")",
                populations,
                population=evidence.split + "; explicit annotated masks only",
                notes=("Empty masks are excluded from positive-truth overlap distributions.",)
                if positive_only
                else (),
            )
        )
    for identifier, title, numerator, denominator in (
        (
            "image_foreground_precision",
            "Per-image foreground precision",
            "true_positive_pixels",
            ("true_positive_pixels", "false_positive_pixels"),
        ),
        (
            "image_foreground_recall",
            "Per-image foreground recall",
            "true_positive_pixels",
            ("true_positive_pixels", "false_negative_pixels"),
        ),
    ):
        rate_populations: list[tuple[str, tuple[float, ...]]] = []
        for name, indices in cohorts:
            values: list[float] = []
            for index in indices:
                record = records[index]
                terms = tuple(record[term] for term in denominator)
                value = record[numerator]
                total = sum(term for term in terms if term is not None)
                if value is not None and total:
                    values.append(value / total)
            rate_populations.append((name, tuple(values)))
        figures.append(
            distribution_figure(
                evidence,
                identifier,
                title,
                "Fraction",
                tuple(rate_populations),
                population=evidence.split + "; per-image rates with eligible denominators",
            )
        )
    for unit in ("px", "m2"):
        for absolute in (False, True):
            name = "area_" + ("absolute" if absolute else "signed") + "_error_" + unit
            error_populations: list[tuple[str, tuple[float, ...]]] = []
            for label, indices in cohorts:
                values = []
                for index in indices:
                    record = records[index]
                    truth, prediction = (
                        record.get("target_area_" + unit),
                        record.get("predicted_area_" + unit),
                    )
                    if truth is not None and prediction is not None:
                        value = prediction - truth
                        if not math.isfinite(value):
                            return Err("captured area difference exceeds finite range")
                        values.append(abs(value) if absolute else value)
                error_populations.append((label, tuple(values)))
            figures.append(
                distribution_figure(
                    evidence,
                    name,
                    "Captured " + name.replace("_", " "),
                    unit,
                    tuple(error_populations),
                    population=evidence.split + "; image-level extent error, not localization",
                )
            )
    spatial = tuple(row.spatial for row in rows if row.spatial is not None)
    for unit in ("px", "m"):
        distances = tuple(
            value
            for record in spatial
            for match in record.localization.matches
            if (value := match.distance_px if unit == "px" else match.distance_m) is not None
        )
        figures.append(
            distribution_figure(
                evidence,
                "matched_centroid_error_" + unit,
                "Conditional matched centroid errors",
                "Centroid error (" + ("pixel" if unit == "px" else "m") + ")",
                (("eligible matched components", distances),),
                unit="COMPONENT",
                population=evidence.split + "; matched components only, misses excluded here",
                notes=(
                    "Read alongside miss-inclusive localization-success and detection coverage.",
                    f"Spatial records captured for {len(spatial)} of {len(rows)} annotated images.",
                ),
            )
        )
    for name in ("asd_px", "hd95_px", "asd_m", "hd95_m"):
        boundary_values = tuple(
            value
            for record in spatial
            if (
                value := record.boundary.asd_px
                if name == "asd_px"
                else record.boundary.hd95_px
                if name == "hd95_px"
                else record.boundary.asd_m
                if name == "asd_m"
                else record.boundary.hd95_m
            )
            is not None
        )
        figures.append(
            distribution_figure(
                evidence,
                "boundary_" + name + "_distribution",
                "Conditional boundary " + name,
                name + " (" + ("m" if name.endswith("_m") else "pixel") + ")",
                (("both nonempty boundaries with geometry", boundary_values),),
                population=evidence.split
                + "; conditional images, empty/missed boundaries excluded",
                notes=(
                    f"Spatial records captured for {len(spatial)} of {len(rows)} annotated images.",
                ),
            )
        )
    expected_curves = (
        "threshold_positive_image_iou",
        "threshold_positive_image_dice",
        "localization_success_px",
        "localization_success_m",
        "pixel_precision_recall_histogram",
        "pixel_roc_histogram",
        "pixel_calibration_reliability",
        "pixel_calibration_support",
    )
    metrics = {metric.name: metric for metric in evidence.metrics}
    for name in dict.fromkeys((*expected_curves, *(curve.name for curve in evidence.curves))):
        metric = metrics.get(
            "foreground_iou_mean_positive_images"
            if name == "threshold_positive_image_iou"
            else "foreground_dice_mean_positive_images"
            if name == "threshold_positive_image_dice"
            else ""
        )
        points = (
            (ModelPoint("captured raw-mask operating point", metric.threshold, metric.value),)
            if metric is not None and metric.threshold is not None and metric.value is not None
            else ()
        )
        style: DrawStyle = (
            "PRE"
            if "precision_recall" in name
            else "POST"
            if name.startswith("localization_success")
            else "BAR"
            if name.endswith("calibration_support")
            else "POINT"
            if name.endswith("calibration_reliability")
            else "LINE"
        )
        figures.append(
            curve_figure(
                evidence,
                name,
                points=points,
                style=style,
                x_range=(0.0, None) if name.startswith("localization_success") else None,
                y_range=(0.0, 1.0) if name.startswith("localization_success") else None,
            )
        )
    figures.extend(metric_figure(evidence, metric) for metric in evidence.metrics)
    if not spatial:
        figures.append(
            ModelFigure(
                "spatial_capture",
                figure_identity(evidence),
                "Spatial evidence coverage",
                "Explicit masks",
                "Captured spatial records",
                evidence.split,
                reason="No compact component/boundary evidence was captured; re-analyze checkpoint",
            )
        )
    return Ok(tuple(figures))
