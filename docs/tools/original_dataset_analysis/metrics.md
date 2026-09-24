# tools.original_dataset_analysis.metrics

**Source:** `packages/tools/src/tools/original_dataset_analysis/metrics.py`
**Kind:** module

## Purpose

This module scores classifier logits and segmentor logit planes on a held-out
batch.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `ClassificationScores` | class | Precision, recall, F1, PR-AUC, ROC-AUC |
| `SegmentationScores` | class | Mean IoU, Dice, accuracy |
| `score_classifier` | function | Scores from logits and labels |
| `score_segmentor` | function | Scores from logit planes and masks |

## Inputs and outputs

`score_classifier(logits, labels) -> ClassificationScores`.

`score_segmentor(logits, masks) -> SegmentationScores`.

## Behavior

1. Probabilities are a sigmoid of the logits. The decision threshold is 0.5.
2. PR-AUC and ROC-AUC use a ranking of those probabilities.
3. Mean IoU averages the positive-class IoU and the background IoU.

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
