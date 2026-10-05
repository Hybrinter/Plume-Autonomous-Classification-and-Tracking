"""Declarative binary metric definitions, directions and limitations.

Definitions are scientific metadata, not a callable dispatch mechanism.
TP/FP/TN/FN refer to the explicitly selected operating point.

Satisfies: REQ-AIML-HIGH-004.
"""

from dataclasses import dataclass
from typing import Literal

from flight.libs.types import Err, Ok, Result


@dataclass(frozen=True, slots=True)
class MetricDefinition:
    """Formula, direction, target population and unavailable-score convention."""

    name: str
    direction: Literal["MINIMIZE", "MAXIMIZE", "DESCRIPTIVE"]
    formula: str
    population: str
    aggregation: str
    undefined_policy: str
    limitations: str
    unit: str = "dimensionless"


_POINT_LIMIT = "Threshold-dependent; report confusion counts, class support and frozen settings."
_NO_DENOMINATOR = "Unavailable when the formula denominator is zero."
_COHORT = "Every eligible image once; no augmentation or training resampling weighting."
_POINT_FORMULAS = (
    ("accuracy", "(TP+TN)/(TP+FP+TN+FN)"),
    ("precision", "TP/(TP+FP)"),
    ("recall", "TP/(TP+FN)"),
    ("specificity", "TN/(TN+FP)"),
    ("negative_predictive_value", "TN/(TN+FN)"),
    ("false_positive_rate", "FP/(FP+TN)"),
    ("false_negative_rate", "FN/(FN+TP)"),
    ("f1", "2*TP/(2*TP+FP+FN)"),
    ("f_beta", "(1+beta^2)*TP/((1+beta^2)*TP+beta^2*FN+FP)"),
    ("balanced_accuracy", "(recall+specificity)/2"),
    ("matthews_correlation", "(TP*TN-FP*FN)/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))"),
    ("predicted_positive_fraction", "(TP+FP)/(TP+FP+TN+FN)"),
)
CLASSIFIER_DEFINITIONS = tuple(
    MetricDefinition(
        name=name,
        direction="MINIMIZE"
        if name in ("false_positive_rate", "false_negative_rate")
        else "DESCRIPTIVE"
        if name == "predicted_positive_fraction"
        else "MAXIMIZE",
        formula=formula,
        population=_COHORT,
        aggregation="confusion_ratio",
        undefined_policy="Unavailable if either truth class is missing."
        if name == "balanced_accuracy"
        else _NO_DENOMINATOR,
        limitations=_POINT_LIMIT,
    )
    for name, formula in _POINT_FORMULAS
) + (
    MetricDefinition(
        name="roc_auc",
        direction="MAXIMIZE",
        formula="Trapezoidal ROC area at complete equal-logit score groups.",
        population=_COHORT,
        aggregation="complete_tie_group_trapezoid",
        undefined_policy="Unavailable if either truth class is missing.",
        limitations="Ranking score, not precision at an operational false-alarm rate.",
    ),
    MetricDefinition(
        name="average_precision",
        direction="MAXIMIZE",
        formula="Sum over complete score groups of (recall_k-recall_previous)*precision_k.",
        population=_COHORT,
        aggregation="complete_tie_group_recall_increments",
        undefined_policy="Unavailable without positive labels; all-positive AP is valid.",
        limitations="Not trapezoidal PR area. Depends on evaluated class prevalence.",
    ),
    MetricDefinition(
        name="binary_cross_entropy",
        direction="MINIMIZE",
        formula="Mean softplus(-abs(z)) + max(z,0) for y=0 or max(-z,0) for y=1.",
        population=_COHORT,
        aggregation="unweighted_image_mean",
        undefined_policy="Infinite probability-baseline loss is unavailable with a reason.",
        limitations="Unweighted probability quality; not a weighted/focal training objective.",
    ),
)
CALIBRATION_DEFINITIONS = (
    MetricDefinition(
        name="brier_score",
        direction="MINIMIZE",
        formula="Mean (p-y)^2.",
        population=_COHORT,
        aggregation="unweighted_probability_mean",
        undefined_policy="Empty cohorts are invalid.",
        limitations="Class prevalence affects the constant-probability baseline.",
    ),
    MetricDefinition(
        name="expected_calibration_error",
        direction="MINIMIZE",
        formula="Sum over occupied bins of (n_bin/N)*abs(mean_probability-positive_fraction).",
        population=_COHORT,
        aggregation="support_weighted_equal_width_bins",
        undefined_policy="Empty bins have null means and zero weight; empty cohorts are invalid.",
        limitations=(
            "Bin-dependent diagnostic; not a proper scoring rule or substitute "
            "for reliability plots."
        ),
    ),
)


def metric_definition(name: str) -> Result[MetricDefinition, str]:
    """Look up an explicit scientific definition, never infer metric direction."""
    for definition in CLASSIFIER_DEFINITIONS + CALIBRATION_DEFINITIONS:
        if definition.name == name:
            return Ok(definition)
    return Err(f"unknown metric definition {name!r}")
