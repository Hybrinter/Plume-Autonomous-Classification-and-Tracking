"""Pure binary classifier scores with complete tied-score ranking groups.

Ranking uses raw logits, never saturated probabilities. Threshold decisions
use log odds, with threshold zero predicting all and one predicting none for
finite logits. Missing denominators remain unavailable rather than zero.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from flight.libs.types import Err, Ok, Result

from tools.ml_models.analysis.config import ScoreConfig
from tools.ml_models.analysis.contracts import (
    AvailabilityRecord,
    CurveEvidence,
    MetricSupport,
    MetricValue,
    NamedCount,
)
from tools.ml_models.analysis.metrics.calibration import CalibrationEvidence, score_calibration
from tools.ml_models.analysis.metrics.inputs import (
    BoolVector,
    FloatVector,
    binary_vectors,
    finite_mean,
    sigmoid,
)


@dataclass(frozen=True, slots=True)
class ConfusionCounts:
    """Exact integer confusion counts at the selected operating point."""

    tp: int
    fp: int
    tn: int
    fn: int


@dataclass(frozen=True, slots=True)
class ClassifierEvidence:
    """Classifier scores and curves, aligned row probabilities and losses."""

    metrics: tuple[MetricValue, ...]
    curves: tuple[CurveEvidence, ...]
    support: MetricSupport
    counts: ConfusionCounts
    probabilities: tuple[float, ...]
    losses: tuple[float | None, ...]
    calibration: CalibrationEvidence
    outputs: tuple[AvailabilityRecord, ...]


def _counts(predicted: BoolVector, target: BoolVector) -> ConfusionCounts:
    return ConfusionCounts(
        tp=int(np.count_nonzero(predicted & target)),
        fp=int(np.count_nonzero(predicted & ~target)),
        tn=int(np.count_nonzero(~predicted & ~target)),
        fn=int(np.count_nonzero(~predicted & target)),
    )


def _metric(
    name: str,
    value: float | None,
    support: MetricSupport,
    *,
    threshold: float | None = None,
    aggregation: str = "confusion_ratio",
    reason: str | None = None,
) -> MetricValue:
    return MetricValue(
        name=name,
        value=value,
        status="AVAILABLE" if value is not None else "UNAVAILABLE",
        reason=None if value is not None else reason or f"{name} has no eligible denominator",
        support=support,
        threshold=threshold,
        aggregation=aggregation,
    )


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _threshold_values(counts: ConfusionCounts, beta: float) -> tuple[tuple[str, float | None], ...]:
    tp, fp, tn, fn = counts.tp, counts.fp, counts.tn, counts.fn
    n = tp + fp + tn + fn
    recall, specificity = _ratio(tp, tp + fn), _ratio(tn, tn + fp)
    if tp == 0:
        f_beta = 0.0 if fp or fn else None
    elif beta >= 1:
        weight = (1 / beta) * (1 / beta)
        f_beta = _ratio((1 + weight) * tp, (1 + weight) * tp + fn + weight * fp)
    else:
        weight = beta * beta
        f_beta = _ratio((1 + weight) * tp, (1 + weight) * tp + weight * fn + fp)
    margin_product = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return (
        ("accuracy", _ratio(tp + tn, n)),
        ("precision", _ratio(tp, tp + fp)),
        ("recall", recall),
        ("specificity", specificity),
        ("negative_predictive_value", _ratio(tn, tn + fn)),
        ("false_positive_rate", _ratio(fp, fp + tn)),
        ("false_negative_rate", _ratio(fn, fn + tp)),
        ("f1", _ratio(2 * tp, 2 * tp + fp + fn)),
        ("f_beta", f_beta),
        (
            "balanced_accuracy",
            (recall + specificity) / 2 if recall is not None and specificity is not None else None,
        ),
        (
            "matthews_correlation",
            (tp * tn - fp * fn) / math.sqrt(margin_product) if margin_product else None,
        ),
        ("predicted_positive_fraction", _ratio(tp + fp, n)),
    )


def _predict(scores: FloatVector, threshold: float, *, logits: bool) -> BoolVector:
    if not logits:
        return scores >= threshold
    if threshold == 0:
        return np.ones(scores.shape, dtype=np.bool_)
    if threshold == 1:
        return np.zeros(scores.shape, dtype=np.bool_)
    return scores >= math.log(threshold) - math.log1p(-threshold)


def _ranking(
    scores: FloatVector,
    target: BoolVector,
    support: MetricSupport,
    *,
    logits: bool,
) -> tuple[tuple[MetricValue, ...], tuple[CurveEvidence, ...], tuple[AvailabilityRecord, ...]]:
    n = len(scores)
    positives = int(target.sum())
    negatives = n - positives
    order = np.argsort(-scores, kind="stable")
    ordered_scores = scores[order]
    ends = np.flatnonzero(np.append(ordered_scores[:-1] != ordered_scores[1:], True))
    tp = np.cumsum(target[order], dtype=np.int64)[ends].astype(np.float64)
    selected = (ends + 1).astype(np.float64)
    fp = selected - tp
    thresholds = (None,) + tuple(float(score) for score in ordered_scores[ends])
    notes = (f"Thresholds are {'raw logits' if logits else 'probabilities'}; None predicts none.",)
    curves: list[CurveEvidence] = []
    outputs: list[AvailabilityRecord] = []
    roc: float | None = None
    ap: float | None = None
    if positives and negatives:
        tpr = np.append(0.0, tp / positives)
        fpr = np.append(0.0, fp / negatives)
        roc = math.fsum(
            float((tpr[index] + tpr[index - 1]) / 2 * (fpr[index] - fpr[index - 1]))
            for index in range(1, len(tpr))
        )
        curves.append(
            CurveEvidence(
                name="roc",
                x_name="false_positive_rate",
                y_name="true_positive_rate",
                x=tuple(float(value) for value in fpr),
                y=tuple(float(value) for value in tpr),
                x_unit="fraction",
                y_unit="fraction",
                support=support,
                thresholds=thresholds,
                notes=notes,
            )
        )
        outputs.append(AvailabilityRecord(name="roc", status="AVAILABLE"))
    else:
        outputs.append(
            AvailabilityRecord(
                name="roc",
                status="UNAVAILABLE",
                reason="ROC requires both truth classes",
            )
        )
    if positives:
        recall = tp / positives
        precision = tp / selected
        ap = math.fsum(
            float(delta * value)
            for delta, value in zip(
                np.diff(np.append(0.0, recall)),
                precision,
                strict=True,
            )
        )
        curves.append(
            CurveEvidence(
                name="precision_recall",
                x_name="recall",
                y_name="precision",
                x=(0.0,) + tuple(float(value) for value in recall),
                y=(1.0,) + tuple(float(value) for value in precision),
                x_unit="fraction",
                y_unit="fraction",
                support=support,
                thresholds=thresholds,
                notes=notes
                + (
                    "The predict-none precision endpoint is a plotting convention, "
                    "not an operating score.",
                    "Use pre-step rendering; AP integrates complete-group precision "
                    "over recall increments.",
                ),
            )
        )
        fraction = selected / n
        for name, values, unit in (
            ("cumulative_gain", recall, "fraction"),
            ("lift", recall / fraction, "ratio"),
        ):
            curves.append(
                CurveEvidence(
                    name=name,
                    x_name="selected_fraction",
                    y_name=name,
                    x=(0.0,) + tuple(float(value) for value in fraction),
                    y=(0.0 if name == "cumulative_gain" else None,)
                    + tuple(float(value) for value in values),
                    x_unit="fraction",
                    y_unit=unit,
                    support=support,
                    thresholds=thresholds,
                    notes=notes,
                )
            )
        outputs.extend(
            AvailabilityRecord(name=name, status="AVAILABLE")
            for name in (
                "precision_recall",
                "cumulative_gain",
                "lift",
            )
        )
    else:
        outputs.extend(
            AvailabilityRecord(
                name=name,
                status="UNAVAILABLE",
                reason="curve requires positive truth examples",
            )
            for name in ("precision_recall", "cumulative_gain", "lift")
        )
    metrics = (
        _metric(
            "roc_auc",
            roc,
            support,
            aggregation="complete_tie_group_trapezoid",
            reason="ROC requires both truth classes",
        ),
        _metric(
            "average_precision",
            ap,
            support,
            aggregation="complete_tie_group_recall_increments",
            reason="AP requires positive truth examples",
        ),
    )
    return metrics, tuple(curves), tuple(outputs)


def _assemble(
    scores: FloatVector,
    probability: FloatVector,
    target: BoolVector,
    losses: tuple[float | None, ...],
    cfg: ScoreConfig,
    *,
    logits: bool,
) -> Result[ClassifierEvidence, str]:
    positives = int(target.sum())
    support = MetricSupport(
        unit="IMAGE",
        n=len(scores),
        counts=(
            NamedCount(name="n_positive", value=positives),
            NamedCount(name="n_negative", value=len(scores) - positives),
        ),
    )
    counts = _counts(_predict(scores, cfg.classifier_probability_threshold, logits=logits), target)
    point_metrics = tuple(
        _metric(
            name,
            value,
            support,
            threshold=cfg.classifier_probability_threshold,
        )
        for name, value in _threshold_values(counts, cfg.f_beta)
    )
    ranking_metrics, ranking_curves, outputs = _ranking(scores, target, support, logits=logits)
    calibration = score_calibration(probability, target, n_bins=cfg.n_calibration_bins)
    if isinstance(calibration, Err):
        return Err(calibration.error)
    bce = (
        finite_mean(np.asarray(losses, dtype=np.float64))
        if all(value is not None for value in losses)
        else None
    )
    loss_metric = _metric(
        "binary_cross_entropy",
        bce,
        support,
        aggregation="unweighted_image_mean",
        reason="baseline assigns zero probability to an observed class; log loss is infinite",
    )
    threshold_records = tuple(
        _threshold_values(_counts(_predict(scores, threshold, logits=logits), target), cfg.f_beta)
        for threshold in cfg.probability_thresholds
    )
    threshold_curves = tuple(
        CurveEvidence(
            name=f"threshold_{name}",
            x_name="probability_threshold",
            y_name=name,
            x=cfg.probability_thresholds,
            y=tuple(dict(record)[name] for record in threshold_records),
            x_unit="probability",
            y_unit="fraction",
            support=support,
        )
        for name, _ in _threshold_values(counts, cfg.f_beta)
    )
    return Ok(
        ClassifierEvidence(
            metrics=point_metrics + ranking_metrics + (loss_metric,) + calibration.value.metrics,
            curves=ranking_curves + threshold_curves + calibration.value.curves,
            support=support,
            counts=counts,
            probabilities=tuple(float(value) for value in probability),
            losses=losses,
            calibration=calibration.value,
            outputs=outputs,
        )
    )


def score_classifier(
    logits: npt.ArrayLike,
    labels: npt.ArrayLike,
    cfg: ScoreConfig | None = None,
) -> Result[ClassifierEvidence, str]:
    """Measure one binary classifier cohort using exact logit ranking."""
    validated = binary_vectors(logits, labels)
    if isinstance(validated, Err):
        return Err(validated.error)
    score, target = validated.value
    losses = np.logaddexp(0.0, -np.abs(score)) + np.where(
        target,
        np.maximum(-score, 0.0),
        np.maximum(score, 0.0),
    )
    return _assemble(
        score,
        sigmoid(score),
        target,
        tuple(float(value) for value in losses),
        cfg if cfg is not None else ScoreConfig(),
        logits=True,
    )


def score_classifier_baseline(
    training_prevalence: float,
    labels: npt.ArrayLike,
    cfg: ScoreConfig | None = None,
) -> Result[ClassifierEvidence, str]:
    """Score a fixed training-derived probability, without fitting held-out labels."""
    if isinstance(training_prevalence, bool) or not isinstance(training_prevalence, (int, float)):
        return Err("training prevalence must be a finite probability")
    if not math.isfinite(training_prevalence) or not 0 <= training_prevalence <= 1:
        return Err("training prevalence must lie in [0, 1]")
    try:
        raw = np.asarray(labels)
    except (TypeError, ValueError) as exc:
        return Err(f"invalid baseline labels: {exc}")
    validated = binary_vectors(np.full(raw.shape, training_prevalence), raw, probabilities=True)
    if isinstance(validated, Err):
        return Err(validated.error)
    probability, target = validated.value
    losses = tuple(
        (-math.log(training_prevalence) if training_prevalence > 0 else None)
        if positive
        else (-math.log1p(-training_prevalence) if training_prevalence < 1 else None)
        for positive in target
    )
    return _assemble(
        probability,
        probability,
        target,
        losses,
        cfg if cfg is not None else ScoreConfig(),
        logits=False,
    )
