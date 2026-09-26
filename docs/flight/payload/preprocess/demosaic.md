# flight.payload.preprocess.demosaic

**Source:** `packages/flight/src/flight/payload/preprocess/demosaic.py`
**Kind:** pure module

## Purpose

This module checks that a prism frame already has one plane per `BAND_ORDER`
entry. The JAI AP-3200T-USB writes three registered RGB planes. There is no
color-filter array to unpack.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `confirm_planes` | function | Accepts `(3, H, W)` and returns float32 |

## Inputs and outputs

`confirm_planes(planes)` takes a candidate stack. Success is float32 of the same
shape. The channel axis must equal `len(BAND_ORDER)`.

## Behavior

1. Reject a rank other than 3.
2. Reject a channel count other than `len(BAND_ORDER)`.
3. Return the stack as float32.

## Errors and faults

| Result | Trigger |
| --- | --- |
| `Err(FRAME_MALFORMED)` | Wrong rank or wrong channel count |

## Messages

None.

## Configuration

None. The channel count comes from `BAND_ORDER`.

## Constraints

Odd height and width are legal. The function does not reorder channels. Cubic
upscale is a later stage.

## Related documents

- [`flight.payload.preprocess`](../preprocess.md)
- [`flight.payload.preprocess.resample`](resample.md)
