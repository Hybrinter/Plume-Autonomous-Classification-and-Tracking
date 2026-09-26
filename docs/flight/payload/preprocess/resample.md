# flight.payload.preprocess.resample

**Source:** `packages/flight/src/flight/payload/preprocess/resample.py`
**Kind:** pure module

## Purpose

This module enlarges a confirmed RGB stack. Height and width grow by an integer
factor. The channel count stays `len(BAND_ORDER)`.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `upsample_planes` | function | Zoom `(3, H, W)` to `(3, factor*H, factor*W)` |

## Inputs and outputs

`upsample_planes(planes, factor, order=3)` takes float planes in `BAND_ORDER`.
`factor` is an integer of at least 1. `order` is 1 (linear) or 3 (cubic).

The result is `Result[np.ndarray, FaultCode]`. Success is float32 of shape
`(3, factor*H, factor*W)`.

## Behavior

1. Reject a stack that is not three planes, a factor below 1, or an order other
   than 1 or 3.
2. Return the input when the factor is 1.
3. For factor 2, double each spatial axis with a separable kernel. Order 1 uses
   the neighbor midpoint. Order 3 uses a Keys cubic kernel.
4. For any other factor, zoom height and width with `scipy.ndimage.zoom`. The
   channel axis is not scaled.

## Errors and faults

| Result | Trigger |
| --- | --- |
| `Err(FRAME_MALFORMED)` | Wrong rank, wrong channel count, factor below 1, or order not 1 or 3 |

## Messages

None.

## Configuration

`PreprocessingConfig.upsample_factor` and `PreprocessingConfig.upsample_order`.

## Constraints

The function is pure. Factor 1 copies the values through. The optical field of
view does not change. The controller uses the upsampled pixel as its IFOV.

## Related documents

- [`flight.payload.preprocess`](../preprocess.md)
- [`flight.payload.preprocess.demosaic`](demosaic.md)
