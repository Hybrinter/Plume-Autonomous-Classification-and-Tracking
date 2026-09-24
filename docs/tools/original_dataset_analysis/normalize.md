# tools.original_dataset_analysis.normalize

**Source:** `packages/tools/src/tools/original_dataset_analysis/normalize.py`
**Kind:** module

## Purpose

This module fits per-channel mean and standard deviation on the stacks a caller
supplies, then scales later stacks with those frozen moments.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `BandStats` | class | Mean and standard deviation |
| `fit_band_stats` | function | Moments from a sequence of stacks |
| `apply_band_stats` | function | Zero-mean unit-variance scaling |

## Inputs and outputs

`fit_band_stats(stacks) -> BandStats`.

`apply_band_stats(stack, stats) -> np.ndarray`.

## Behavior

1. Moments pool every pixel of every supplied stack, per channel.
2. Standard deviation is floored at ``1e-6``.
3. Scaling subtracts the mean and divides by the standard deviation.

## Errors and faults

`ValueError` when the stack list is empty, channel counts differ, or a stack
does not match the stored channel count.

## Messages

None.

## Configuration

None.

## Constraints

The caller passes train stacks only. This module does not open archives.

## Related documents

- [`tools.original_dataset_analysis`](../original_dataset_analysis.md)
- [`tools.original_dataset_analysis.dataset`](dataset.md)
