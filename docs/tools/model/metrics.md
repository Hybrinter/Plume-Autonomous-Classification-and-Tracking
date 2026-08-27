# tools.model.metrics

**Source:** `packages/tools/src/tools/model/metrics.py`
**Kind:** pure module

## Purpose

This module scores binary presence predictions for the classifier. Mask IoU lives
in `tools.model.accept`.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `binary_accuracy` | function | 1.0 on a matching boolean pair, else 0.0 |
| `mean_binary_accuracy` | function | Mean accuracy over aligned pairs |

## Inputs and outputs

`binary_accuracy(pred_positive, label_positive) -> float`.

`mean_binary_accuracy(pred_positive, label_positive) -> float` in [0, 1]. Empty
inputs return 0.0. Unequal lengths raise `ValueError`.

## Behavior

1. `binary_accuracy` compares two booleans with identity.
2. `mean_binary_accuracy` averages `binary_accuracy` over zip-strict pairs.

## Errors and faults

`ValueError` when prediction and label tuples differ in length.

## Messages

None.

## Configuration

None.

## Constraints

Pure Python. No torch. No file I/O.

## Related documents

- [`tools.model`](../model.md)
- [`tools.model.accept`](accept.md)
