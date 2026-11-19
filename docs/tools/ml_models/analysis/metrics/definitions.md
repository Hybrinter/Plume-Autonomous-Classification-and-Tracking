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
| `SEGMENTATION_DEFINITIONS` | constant | Definitions for the segmentation overlap, area, loss, and pixel diagnostics |
| `LOCALIZATION_DEFINITIONS` | constant | Definitions for component counts, match rates, and conditional centroid errors |
| `BOUNDARY_DEFINITIONS` | constant | Definitions for boundary hit rates, image counts, and conditional distances |
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
classifier and calibration definition declares the cohort population:
every eligible image once, with no augmentation or training-resampling
weighting.

`SEGMENTATION_DEFINITIONS` covers the overlap means (per-image
`TP/(TP+FP+FN)` and Dice over nonempty-truth and all-annotated images,
with empty/empty defined as one), pooled global confusion ratios,
per-image and pixel-weighted BCE and Brier means, area errors in pixels
and in square metres by local GSD, verified-negative foreground/blob
rates, truth-class Brier and probability-residual means, the approximate
histogram pixel ROC and AP, and the exact occupied-bin pixel ECE.
Overlap, precision, recall, and the pixel ranking scores maximize;
signed area errors and probability residuals are descriptive; the rest
minimize. Shared policies state that metrics are unavailable when no
eligible observations or a zero denominator exists, ROC needs both truth
classes, and AP needs positives; limitations record correlated pixels,
the local-GSD area approximation, the exclusion of empty images from
positive-truth headlines, and approximate histogram ranking with
explicit thresholds and aggregation.

`LOCALIZATION_DEFINITIONS` covers exact component counts
(`truth_components`, `predicted_components`, matched and unmatched
populations), the pooled match rates `component_precision`,
`component_recall`, and `component_f1`, split/merge component counts,
and the conditional `matched_centroid_error_{px,m}_{mean,median,p95}`
metrics. Precision, recall, and F1 maximize; counts are descriptive;
centroid errors minimize. `BOUNDARY_DEFINITIONS` covers the pooled hit
rates `boundary_precision`, `boundary_recall`, and `boundary_f1`, the
exact missed/spurious/empty image counts, and the conditional
equal-image `boundary_{asd,hd95}_{px,m}_mean` distances. Hit rates
maximize; image counts are descriptive; distances minimize. Both
families declare that zero-denominator rates are unavailable, that
metre geometry requires recorded GSD, and that conditional distances
exclude detection misses, which the separate image counts and
miss-inclusive success curves carry.

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
- [`tools.ml_models.analysis.metrics.spatial`](spatial.md)
