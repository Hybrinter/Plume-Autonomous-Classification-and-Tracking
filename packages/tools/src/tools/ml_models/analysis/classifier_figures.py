"""Classifier figures from frozen split curves and complete compact scalar capture.

Ranking, calibration and threshold coordinates pass through unchanged.
Confusion displays use captured error flags at the recorded operating point;
distributions reduce recorded scalars once. No threshold is fitted, no
probability is recalibrated, and no source or model is read.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

from typing import cast

from flight.libs.types import Err, Ok, Result

from tools.ml_models.analysis.capture import CaptureRow
from tools.ml_models.analysis.contracts import SplitEvidence
from tools.ml_models.analysis.model_figures import (
    DrawStyle,
    ModelFigure,
    ModelPoint,
    ModelSeries,
    captured_values,
    curve_figure,
    distribution_figure,
    figure_identity,
    metric_figure,
    validate_figure_rows,
)


def classifier_figure_data(
    evidence: SplitEvidence,
    rows: tuple[CaptureRow, ...],
) -> Result[tuple[ModelFigure, ...], str]:
    """Freeze complete classifier figure inventory; unsupported populations stay unavailable."""
    if evidence.task != "classifier":
        return Err("classifier figures require classifier evidence")
    checked = validate_figure_rows(evidence, rows)
    if isinstance(checked, Err):
        return checked
    metrics = {metric.name: metric for metric in evidence.metrics}
    required = tuple(name for name in ("logit", "probability", "binary_cross_entropy"))
    records = tuple(captured_values(row) for row in rows)
    if any(any(record.get(name) is None for name in required) for record in records):
        return Err("classifier distributions require complete captured logits/probabilities/losses")
    if any(
        row.false_positive
        and row.false_negative
        or row.false_positive
        and row.label != 0
        or row.false_negative
        and row.label != 1
        for row in rows
    ):
        return Err("captured classifier error flags conflict with truth labels")
    if any(
        value is not None and not 0 <= value <= 1
        for record in records
        for value in (record.get("probability"),)
    ):
        return Err("captured classifier probabilities must lie in [0,1]")
    n = len(rows)
    positive = sum(row.label == 1 for row in rows)
    negative = n - positive
    fp = sum(row.false_positive for row in rows)
    fn = sum(row.false_negative for row in rows)
    tp, tn = positive - fn, negative - fp
    support = evidence.support
    prevalence = positive / n if n else None
    identity = figure_identity(evidence)
    threshold_metric = metrics.get("precision")
    threshold = threshold_metric.threshold if threshold_metric else None
    operating = f"Captured operating threshold: {threshold!r}"
    expected_ratios = (
        ("precision", tp / (tp + fp) if tp + fp else None),
        ("recall", tp / positive if positive else None),
        ("false_positive_rate", fp / negative if negative else None),
        ("accuracy", (tp + tn) / n if n else None),
    )
    for name, expected in expected_ratios:
        metric = metrics.get(name)
        if metric is not None and (
            (metric.value is None) != (expected is None)
            or metric.value is not None
            and expected is not None
            and abs(metric.value - expected) > 1e-12
        ):
            return Err("captured classifier flags disagree with the frozen operating metrics")

    def point(x_name: str, y_name: str) -> tuple[ModelPoint, ...]:
        """Copy a single recorded operating point only when both coordinates exist."""
        x, y = metrics.get(x_name), metrics.get(y_name)
        return (
            (ModelPoint(operating, x.value, y.value),)
            if x is not None and y is not None and x.value is not None and y.value is not None
            else ()
        )

    figures: list[ModelFigure] = []
    for name, baseline, points, style, reason in (
        (
            "precision_recall",
            ModelSeries(
                "recorded cohort prevalence", (0.0, 1.0), (prevalence, prevalence), support
            ),
            point("recall", "precision"),
            "PRE",
            "PR requires positive truth examples",
        ),
        (
            "roc",
            ModelSeries("chance diagonal", (0.0, 1.0), (0.0, 1.0), support),
            point("false_positive_rate", "recall"),
            "LINE",
            "ROC requires both truth classes",
        ),
        (
            "cumulative_gain",
            ModelSeries("random-selection diagonal", (0.0, 1.0), (0.0, 1.0), support),
            (),
            "LINE",
            "Gains require positive truth examples",
        ),
        (
            "lift",
            ModelSeries("random-selection lift", (0.0, 1.0), (1.0, 1.0), support),
            (),
            "LINE",
            "Lift requires positive truth examples",
        ),
        (
            "calibration_reliability",
            ModelSeries("perfect calibration", (0.0, 1.0), (0.0, 1.0), support),
            (),
            "POINT",
            None,
        ),
        ("calibration_support", None, (), "BAR", None),
    ):
        figures.append(
            curve_figure(
                evidence,
                name,
                baseline=baseline,
                points=points,
                style=cast("DrawStyle", style),
                reason=reason,
            )
        )
    for curve in evidence.curves:
        if curve.name.startswith("threshold_"):
            metric = metrics.get(curve.name.removeprefix("threshold_"))
            points = (
                (ModelPoint(operating, metric.threshold, metric.value),)
                if metric is not None and metric.threshold is not None and metric.value is not None
                else ()
            )
            figures.append(curve_figure(evidence, curve.name, points=points))
    matrix = ((float(tn), float(fp)), (float(fn), float(tp)))
    matrix_counts = ((tn, fp), (fn, tp))
    for name, values, label, bounds in (
        ("confusion_counts", matrix, "Count (images)", None),
        (
            "confusion_truth_normalized",
            tuple(
                tuple(value / total if total else None for value in row)
                for row, total in zip(matrix, (negative, positive), strict=True)
            ),
            "Fraction within truth class",
            (0.0, 1.0),
        ),
        (
            "confusion_prediction_normalized",
            tuple(
                tuple(
                    value / total if total else None
                    for value, total in zip(row, (tn + fn, fp + tp), strict=True)
                )
                for row in matrix
            ),
            "Fraction within prediction class",
            (0.0, 1.0),
        ),
    ):
        figures.append(
            ModelFigure(
                name,
                identity,
                "Captured " + name.replace("_", " "),
                "Predicted class",
                "Truth class",
                evidence.split + " images; " + label,
                kind="MATRIX",
                x_categories=("negative", "positive"),
                y_categories=("negative", "positive"),
                matrix=values,
                matrix_support=matrix_counts,
                matrix_range=bounds,
                reason=None if n else "No captured classifier images",
                notes=(operating,),
            )
        )
    figures.append(
        ModelFigure(
            "class_counts",
            identity,
            "Captured classifier truth counts",
            "Truth class",
            "Count (images)",
            evidence.split + " complete recorded image cohort",
            (
                ModelSeries(
                    "images", (0.0, 1.0), (float(negative), float(positive)), support, "BAR"
                ),
            ),
            x_categories=("negative", "positive"),
        )
    )
    for name, label in (
        ("logit", "Raw classifier logit"),
        ("probability", "Positive-class probability"),
        ("binary_cross_entropy", "Unweighted per-image BCE"),
        ("objective_loss", "Configured weighted objective"),
    ):
        populations = tuple(
            (
                class_name,
                tuple(
                    value
                    for row, record in zip(rows, records, strict=True)
                    if row.label == target and (value := record.get(name)) is not None
                ),
            )
            for class_name, target in (("negative truth", 0), ("positive truth", 1))
        )
        figures.append(
            distribution_figure(
                evidence,
                name + "_distribution",
                "Class-conditioned " + label,
                label,
                populations,
                population=evidence.split + " captured images, grouped by truth class",
            )
        )
    confidence: list[tuple[bool, float]] = []
    for row, record in zip(rows, records, strict=True):
        probability = record["probability"]
        if probability is None:
            continue
        predicted_positive = row.false_positive or row.label == 1 and not row.false_negative
        confidence.append(
            (
                not (row.false_positive or row.false_negative),
                probability if predicted_positive else 1 - probability,
            )
        )
    figures.append(
        distribution_figure(
            evidence,
            "prediction_confidence",
            "Correct/incorrect predicted-class confidence",
            "Probability assigned to the predicted class",
            tuple(
                (label, tuple(value for is_correct, value in confidence if is_correct == correct))
                for label, correct in (("correct", True), ("incorrect", False))
            ),
            population=evidence.split + " captured decisions at the recorded threshold",
            notes=(
                "This is predicted-class probability, not max(p, 1-p); non-0.5 "
                "thresholds can select the less probable class.",
            ),
        )
    )
    figures.extend(metric_figure(evidence, metric) for metric in evidence.metrics)
    return Ok(tuple(figures))
