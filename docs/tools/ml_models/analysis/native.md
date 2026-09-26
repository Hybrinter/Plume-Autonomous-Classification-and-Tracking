# tools.ml_models.analysis.native

**Source:** `packages/tools/src/tools/ml_models/analysis/native.py`
**Kind:** module

## Purpose

This module expands a coarse segmentor logit plane onto the 120 px mask and
scores Dice and positive-class IoU.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `NativeGridScores` | class | `native_dice` and positive IoU |
| `score_on_native_grid` | function | Nearest upsample, then overlap scores |

## Inputs and outputs

`score_on_native_grid(logits, native_masks) -> NativeGridScores`.

`logits` has shape `(N, 1, H, W)`. `native_masks` has shape `(N, 120, 120)`
or `(N, 1, 120, 120)`.

## Behavior

1. Repeat each coarse logit onto the 120 px grid with nearest interpolation.
2. Threshold the sigmoid at 0.5.
3. Return positive-class Dice as `dice` and positive-class IoU as
   `positive_iou`. The ground-sample chart labels this Dice `native_dice`.

## Errors and faults

`ValueError` when the batch counts differ or the mask is not 120 px.

## Messages

None.

## Configuration

The native side is 120 px. The decision threshold is 0.5.

## Constraints

This module imports torch. It does not import `flight.payload.inference`.

## Related documents

- [`tools.ml_models.analysis`](../analysis.md)
- [`tools.ml_models.analysis.plots`](plots.md)
