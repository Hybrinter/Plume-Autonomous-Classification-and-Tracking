# tools.ml_models.analysis.classifier_figures

**Source:** `packages/tools/src/tools/ml_models/analysis/classifier_figures.py`
**Kind:** module

## Purpose

Classifier figures from frozen split curves and complete compact
scalar capture. Ranking, calibration, and threshold coordinates pass
through unchanged. Confusion displays use captured error flags at the
recorded operating point; distributions reduce recorded scalars once.
No threshold is fitted, no probability is recalibrated, and no source
or model is read.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `classifier_figure_data` | function | Freeze the complete classifier figure inventory |

## Inputs and outputs

`classifier_figure_data(evidence, rows) ->
Result[tuple[ModelFigure, ...], str]` requires classifier
`SplitEvidence` and the complete captured-row cohort.

## Behavior

Emitted figures include `precision_recall` (PRE step with prevalence
baseline), `roc`, `cumulative_gain`, `lift`,
`calibration_reliability`, `calibration_support`, every captured
`threshold_*` curve with its recorded operating point, three confusion
variants (`confusion_counts`, `confusion_truth_normalized`,
`confusion_prediction_normalized`), `class_counts`, class-conditioned
`logit`/`probability`/`binary_cross_entropy`/`objective_loss`
distributions, `prediction_confidence` (predicted-class probability at
the recorded threshold), and one `metric_*` figure per captured
metric. Unsupported populations keep explicit `reason` unavailability;
no coordinates are fabricated.

## Errors and faults

Returns `Err` for non-classifier evidence, incomplete captured
logits/probabilities/losses, error flags conflicting with truth
labels, out-of-range probabilities, or captured flags disagreeing
with the frozen operating metrics.

## Messages

None.

## Configuration

None.

## Constraints

- Confusion denominators come from the captured truth classes; empty
  classes produce null cells, not fabricated fractions.
- The confidence distribution uses the actual predicted class at the
  recorded threshold.

## Related documents

- [`tools.ml_models.analysis`](../analysis.md)
- [`tools.ml_models.analysis.model_figures`](model_figures.md)
- [`tools.ml_models.analysis.plots.classifier`](plots/classifier.md)
