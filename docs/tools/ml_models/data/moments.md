# tools.ml_models.data.moments

**Source:** `packages/tools/src/tools/ml_models/data/moments.py`
**Kind:** module

## Purpose

This module fits per-channel mean and standard deviation on the stacks a caller
supplies, then scales later stacks with those frozen moments.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `BandStats` | class | Mean and standard deviation |
| `MomentAccumulator` | class | Running pixel sums |
| `fit_band_stats` | function | Moments from a sequence of stacks |
| `check_band_stats` | function | Require shape ``(C,)`` for both vectors |
| `apply_band_stats` | function | Zero-mean unit-variance scaling |

## Inputs and outputs

`fit_band_stats(stacks) -> BandStats`.

`check_band_stats(stats, channels) -> None`.

`apply_band_stats(stack, stats) -> np.ndarray`.

## Behavior

1. ``MomentAccumulator.update`` adds one stack. ``finish`` returns the same
   moments as ``fit_band_stats`` on the same stacks. Both pool every pixel.
2. Standard deviation is floored at ``1e-6``.
3. Scaling subtracts the mean and divides by the standard deviation. Both
   vectors must have shape ``(C,)``.

## Errors and faults

`ValueError` when the stack list is empty, channel counts differ, or either
moment vector is not exactly shape ``(C,)``.

## Messages

None.

## Configuration

None.

## Constraints

The caller passes train stacks only. This module does not open archives.

## Related documents

- [`tools.ml_models.data`](../data.md)
- [`tools.ml_models.data.norm`](norm.md)
- [`tools.ml_models.train.tiles`](../../ml_models/train/tiles.md)
