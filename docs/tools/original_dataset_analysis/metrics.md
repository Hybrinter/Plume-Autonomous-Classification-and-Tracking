# tools.original_dataset_analysis.metrics

**Source:** `packages/tools/src/tools/original_dataset_analysis/metrics.py`
**Kind:** module

## Purpose

This module scores classifier logits and segmentor logit planes on a held-out
batch. `score_on_native_grid` is re-exported from `tools.ml_models.analysis.native`.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `ClassificationScores` | class | Precision, recall, F1, PR-AUC, ROC-AUC |
| `SegmentationScores` | class | Mean IoU, Dice, accuracy |
| `score_classifier` | function | Scores from logits and labels |
| `score_segmentor` | function | Scores from logit planes and masks |
| `NativeGridScores` | class | Dice and positive IoU on the 120 px mask |
| `score_on_native_grid` | function | Coarse logits expanded onto the native mask |

## Inputs and outputs

`score_classifier(logits, labels) -> ClassificationScores`.

`score_segmentor(logits, masks) -> SegmentationScores`.

`score_on_native_grid(logits, native_masks) -> NativeGridScores`.

## Behavior

1. Probabilities are a sigmoid of the logits. The decision threshold is 0.5.
2. PR-AUC and ROC-AUC rank the logits. Equal logits form one threshold.
3. Mean IoU averages the positive-class IoU and the background IoU.
4. ``score_on_native_grid`` repeats each coarse logit across the native cells
   it covers, then scores Dice and positive-class IoU on the 120 px mask.

## Errors and faults

`ValueError` when the logit and target shapes differ.

## Messages

None.

## Configuration

None.

## Constraints

A split with only one class returns ranking areas of zero.

## Related documents

- [`tools.original_dataset_analysis`](../original_dataset_analysis.md)
- [`tools.original_dataset_analysis.train`](train.md)
