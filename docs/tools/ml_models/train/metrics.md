# tools.ml_models.train.metrics

**Source:** `packages/tools/src/tools/ml_models/train/metrics.py`
**Kind:** module

## Purpose

This module scores classifier logits and segmentor masks: accuracy,
precision, recall, F1, ROC and PR areas, Brier, BCE, IoU, Dice, and a
blob-gate IoU at a stricter threshold.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `BLOB_GATE` | constant | Segmentor gate threshold 0.55 |
| `MASK_THRESHOLD` | constant | Mask binarisation threshold 0.5 |
| `LOGIT_THRESHOLD` | constant | Classifier logit threshold 0.0 |
| `ConfusionCounts` | dataclass | TP/FP/TN/FN counts |
| `ReliabilityBin` | dataclass | Calibration bucket statistics |
| `ClassifierMetrics` | dataclass | Accuracy through BCE score block |
| `SegmentorMetrics` | dataclass | IoU, Dice, gate IoU, and BCE block |
| `sigmoid` | function | Logits to probabilities |
| `binary_accuracy` | function | Single-sample accuracy |
| `mean_binary_accuracy` | function | Batch accuracy |
| `compute_iou` | function | Mask IoU at a threshold |
| `compute_dice` | function | Mask Dice at a threshold |
| `confusion_counts` | function | Thresholded confusion counts |
| `precision_recall_f1` | function | Scores from counts |
| `roc_auc` | function | Rank-based ROC area |
| `average_precision` | function | PR area |
| `brier_score` | function | Probability calibration score |
| `reliability_bins` | function | Calibration buckets |
| `binary_cross_entropy_with_logits` | function | Mean BCE on logits |
| `classifier_metrics` | function | Full classifier metric block |
| `segmentor_metrics` | function | Full segmentor metric block |

## Inputs and outputs

Helpers accept array-likes and return Python floats or small dataclasses.
`classifier_metrics(logits, targets)` returns `ClassifierMetrics`;
`segmentor_metrics(logits, targets)` returns `SegmentorMetrics`.

## Behavior

1. Classifier scores threshold probabilities at `LOGIT_THRESHOLD` on the
   logit; ROC and PR areas use ranking, not thresholding.
2. Segmentor scores binarise probabilities at `MASK_THRESHOLD`; the blob
   gate rescores IoU at `BLOB_GATE`.
3. Empty-class ratios return 0.0 without failing.

## Errors and faults

`ValueError` on mismatched prediction/label shapes in the block
functions.

## Messages

None.

## Configuration

Thresholds are module constants; `val_metric` in `TrainConfig` selects
which reported metric drives best-checkpoint selection.

## Constraints

This module imports torch and numpy at import time.

## Related documents

- [`tools.ml_models.train`](../train.md)
- [`tools.ml_models.train.evaluate`](evaluate.md)
- [`tools.ml_models.train.config`](config.md)
