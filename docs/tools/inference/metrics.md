# tools.inference.metrics

**Source:** `packages/tools/src/tools/inference/metrics.py`
**Kind:** pure module

## Purpose

This module scores classifier presence predictions and segmentor masks. It stays
torch-free.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `binary_accuracy` | function | 1.0 on a matching boolean pair, else 0.0 |
| `mean_binary_accuracy` | function | Mean accuracy over aligned pairs |
| `compute_iou` | function | Binary mask IoU at a probability threshold |
| `compute_dice` | function | Binary mask Dice at a probability threshold |
| `confusion_counts` | function | tp, fp, tn, fn |
| `precision_recall_f1` | function | Precision, recall, and F1 from counts |
| `roc_auc` | function | ROC area from scores and labels |
| `average_precision` | function | Area under the precision-recall curve |
| `brier_score` | function | Mean squared probability error |
| `reliability_bins` | function | Calibration bins |
| `classifier_metrics` | function | Split-level classifier summary |
| `segmentor_metrics` | function | Split-level mask overlap summary |
| `binary_cross_entropy_with_logits` | function | Mean BCE on logits |
| `ClassifierMetrics` | class | Classifier summary fields |
| `SegmentorMetrics` | class | Segmentor summary fields |

## Inputs and outputs

`binary_accuracy(pred_positive, label_positive) -> float`.

`mean_binary_accuracy(pred_positive, label_positive) -> float` in [0, 1]. Empty
inputs return 0.0. Unequal lengths raise `ValueError`.

`compute_iou(pred_mask, gold_mask, threshold=0.5) -> float` in [0, 1]. Two empty
masks score 1.0.

`classifier_metrics(logits, labels, logit_threshold=0.0) -> ClassifierMetrics`.
n=0 yields zeros.

`segmentor_metrics(logits, masks, mask_threshold=0.5, blob_gate=0.55) -> SegmentorMetrics`.
n=0 yields zeros.

## Behavior

1. `binary_accuracy` compares two booleans with identity.
2. `mean_binary_accuracy` averages `binary_accuracy` over zip-strict pairs.
3. Classifier threshold defaults to logit 0.0 (probability 0.5).
4. Segmentor IoU and Dice use threshold 0.5. A second IoU uses blob gate 0.55.
5. ROC, PR, and Brier consume sigmoid probabilities of the logits.
6. Empty splits return 0.0 with `n=0`. ROC-AUC is 0.0 when a class is missing.

## Errors and faults

`ValueError` when prediction and label lengths or ranks do not match.

## Messages

None.

## Configuration

`LOGIT_THRESHOLD` is 0.0. `MASK_THRESHOLD` is 0.5. `BLOB_GATE` is 0.55.

## Constraints

Pure NumPy. No torch. No file I/O.

## Related documents

- [`tools.inference`](../inference.md)
- [`tools.inference.accept`](accept.md)
