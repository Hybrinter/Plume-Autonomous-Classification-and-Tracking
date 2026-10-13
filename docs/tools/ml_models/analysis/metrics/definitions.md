# tools.ml_models.analysis.metrics.definitions

**Source:** `packages/tools/src/tools/ml_models/analysis/metrics/definitions.py`
**Kind:** module

## Purpose

Declarative binary metric definitions, directions, and limitations.
Definitions are scientific metadata, not a callable dispatch mechanism.
TP/FP/TN/FN refer to the explicitly selected operating point.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `MetricDefinition` | dataclass | Name, direction, formula, population, aggregation, undefined policy, limitations, unit |
| `CLASSIFIER_DEFINITIONS` | constant | Definitions for the classifier point, ranking, and loss metrics |
| `CALIBRATION_DEFINITIONS` | constant | Definitions for Brier and ECE |
| `metric_definition` | function | Name lookup; `Result` boundary |

## Inputs and outputs

`metric_definition` takes a metric name and returns
`Result[MetricDefinition, str]`; unknown names return `Err`.

## Behavior

`CLASSIFIER_DEFINITIONS` covers the twelve operating-point ratios
(`accuracy` through `predicted_positive_fraction`, with `f_beta` using
the `(1+beta^2)*TP/((1+beta^2)*TP+beta^2*FN+FP)` formula), `roc_auc`
(trapezoidal area at complete equal-logit score groups),
`average_precision` (sum over complete score groups of
`(recall_k - recall_previous) * precision_k`), and
`binary_cross_entropy` (mean softplus loss per row). False rates
minimize, `predicted_positive_fraction` is descriptive, and the rest
maximize; `binary_cross_entropy` minimizes. Each point metric declares
that it is threshold-dependent and becomes unavailable when its
denominator is zero (`balanced_accuracy` additionally requires both
truth classes). `roc_auc` is unavailable if either truth class is
missing, `average_precision` is unavailable without positive labels
(all-positive AP is valid), and infinite probability-baseline loss is
unavailable with a reason. `CALIBRATION_DEFINITIONS` covers
`brier_score` (mean `(p - y) ** 2`) and `expected_calibration_error`
(support-weighted absolute gaps over occupied equal-width bins). Every
definition declares the cohort population: every eligible image once,
with no augmentation or training-resampling weighting.

## Errors and faults

`metric_definition` returns `Err` for unknown names; directions are
never inferred.

## Messages

None.

## Configuration

None.

## Constraints

- These records are metadata for reports and registries; they contain no
  executable scoring code.
- AP is not trapezoidal precision-recall area; ECE is bin-dependent and
  not a proper scoring rule; BCE is unweighted probability quality, not
  a weighted or focal training objective.

## Related documents

- [`tools.ml_models.analysis.metrics`](../metrics.md)
- [`tools.ml_models.analysis.metrics.classifier`](classifier.md)
- [`tools.ml_models.analysis.metrics.calibration`](calibration.md)
