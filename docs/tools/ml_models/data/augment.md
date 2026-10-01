# tools.ml_models.data.augment

**Source:** `packages/tools/src/tools/ml_models/data/augment.py`
**Kind:** module

## Purpose

This module applies one geometric transform to an image and its mask.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `dihedral` | function | Flip and quarter-turns on image and mask |

## Inputs and outputs

`dihedral(image, mask, k) -> tuple[np.ndarray, np.ndarray]`.

`image` is `(C, H, W)`. `mask` is `(1, H, W)`. `k` is an int in `0..7`.

## Behavior

1. `k % 4` is the number of counterclockwise quarter turns. `k >= 4` flips
   the width axis before those turns. The image and the mask receive the
   same transform. The results are contiguous float32 arrays.

## Errors and faults

`ValueError` when `k` is outside `0..7`, or when the image and mask shapes
disagree.

## Messages

None.

## Configuration

None.

## Constraints

The mask channel count is 1.

## Related documents

- [`tools.ml_models.data`](../data.md)
