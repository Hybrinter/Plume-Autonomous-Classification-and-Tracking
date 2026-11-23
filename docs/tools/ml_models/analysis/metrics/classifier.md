# tools.ml_models.analysis.metrics.classifier

**Source:** `packages/tools/src/tools/ml_models/analysis/metrics/classifier.py`
**Kind:** module

## Purpose

Pure binary classifier scores with complete tied-score ranking groups.
Ranking uses raw logits, never saturated probabilities. Threshold
decisions use log odds, with a threshold of zero predicting all and one
predicting none for finite logits. Missing denominators stay
unavailable; they never report zero.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `ConfusionCounts` | dataclass | Exact `tp`/`fp`/`tn`/`fn` at the selected operating point |
| `ClassifierEvidence` | dataclass | Metrics, curves, support, counts, probabilities, losses, calibration, output availability |
| `score_classifier` | function | Scores one binary cohort from raw logits; `Result` boundary |
| `score_classifier_baseline` | function | Scores a fixed training-derived probability; `Result` boundary |

## Inputs and outputs

`score_classifier` takes logit and label array-likes plus an optional
`ScoreConfig` and returns `Result[ClassifierEvidence, str]`.
`score_classifier_baseline` takes a finite `training_prevalence` in
`[0, 1]`, label array-likes, and an optional `ScoreConfig`, returning the
same `Result` shape.

## Behavior

Confusion counts come from `scores >= log-odds(threshold)` for logits,
or `p >= threshold` for probabilities. Point metrics (accuracy through
`predicted_positive_fraction`, per `definitions`) evaluate at
`classifier_probability_threshold`. Ranking sorts raw scores descending
and evaluates only at complete tied-score groups: ROC is a trapezoid
over those groups, and average precision sums
`(recall_k - recall_previous) * precision_k` over complete groups — it is
not trapezoidal precision-recall area. The precision-recall curve prepends
a `(0, 1)` endpoint as a plotting convention only, and its notes direct
pre-step rendering. Cumulative-gain and lift curves share the same
thresholds; lift has a null origin. ROC is unavailable when either truth
class is missing; the precision-recall, gain, and lift curves are
unavailable without positive labels, while all-positive average
precision remains valid. Binary cross-entropy is the unweighted mean of
per-row losses; when any loss is null (a probability baseline assigning
zero to an observed class) the metric is unavailable, never clipped. Threshold curves evaluate every point metric across the
configured probability grid. The baseline path scores one fixed
probability with `p >= threshold` decisions and computes its losses
directly, producing null losses exactly where the log loss is infinite.

## Errors and faults

Invalid vectors propagate `Err` from `binary_vectors`. A non-finite or
out-of-range `training_prevalence` returns `Err`. Calibration failures
propagate `Err`. Unavailable metrics and curves are explicit
`MetricValue`/`AvailabilityRecord` states, never fabricated scores.

## Messages

None.

## Configuration

`ScoreConfig` supplies `classifier_probability_threshold`, `f_beta`,
`n_calibration_bins`, and `probability_thresholds`; defaults apply when
no config is passed.

## Constraints

- Ranking never uses sigmoid-saturated probabilities; raw logits keep
  distinct tied groups.
- The predict-none precision endpoint is a rendering convention, not an
  operating score.
- Infinite baseline log loss is reported as unavailable, not clipped.
- The baseline scorer does not fit or reweight held-out labels.

## Related documents

- [`tools.ml_models.analysis.metrics`](../metrics.md)
- [`tools.ml_models.analysis.metrics.definitions`](definitions.md)
- [`tools.ml_models.analysis.metrics.calibration`](calibration.md)
- [`tools.ml_models.analysis.metrics.inputs`](inputs.md)
