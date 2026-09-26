# tools.ml_models.data.augment

**Source:** `packages/tools/src/tools/ml_models/data/augment.py`
**Kind:** module

## Purpose

This module applies one geometric transform to an image and its mask, and
blends a chip onto a canvas.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `dihedral` | function | Flip and quarter-turns on image and mask |
| `overlap_window` | function | Overlapping source and destination rectangles |
| `feather_paste` | function | In-place border blend |

## Inputs and outputs

`dihedral(image, mask, k) -> tuple[np.ndarray, np.ndarray]`.

`image` is `(C, H, W)`. `mask` is `(1, H, W)`. `k` is an int in `0..7`.

`overlap_window(canvas_hw, chip_hw, top, left) -> tuple | None`.

`feather_paste(canvas, chip, top, left, feather_px) -> None`.

## Behavior

1. `k % 4` is the number of counterclockwise quarter turns. `k >= 4` flips
   the width axis before those turns. The image and the mask receive the
   same transform. The results are contiguous float32 arrays.
2. `overlap_window` clips a chip whose origin is negative or whose far edge
   passes the canvas. It returns `None` when the chip misses the canvas.
3. `feather_paste` writes only the overlap. Pixels at least `feather_px`
   from the chip edge replace the canvas. Pixels in the outer `feather_px`
   band use `alpha * chip + (1 - alpha) * canvas`. `alpha` is
   `(inset + 1) / (feather_px + 1)`, where `inset` is the distance to the
   nearest chip edge. `feather_px` of 0 replaces every overlapping pixel.
4. `feather_paste` does not read or write a mask.

## Errors and faults

`ValueError` when `k` is outside `0..7`, when the image and mask shapes
disagree, when the canvas and chip channel counts differ, or when
`feather_px` is negative.

## Messages

None.

## Configuration

None.

## Constraints

The mask channel count is 1. The paste updates the canvas in place.

## Related documents

- [`tools.ml_models.data`](../data.md)
- [`tools.ml_models.data.canvas`](canvas.md)
