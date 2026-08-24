# flight.payload.preprocess.band_select

**Source:** `packages/flight/src/flight/payload/preprocess/band_select.py`
**Kind:** pure module

## Purpose

This module reorders demosaicked band planes from sensor layout order into the channel
order the inference model expects.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `select_bands` | function | Gathers planes by band name into model input order |

## Inputs and outputs

`select_bands(planes, layout, band_names)` takes `(len(layout), H, W)` planes in
mosaic cell order. It returns `Result[np.ndarray, FaultCode]` with shape
`(len(band_names), H, W)`.

## Behavior

1. Verify the plane count matches `layout` length and the array is 3-D.
2. Resolve each name in `band_names` to an index in `layout`.
3. Return the gathered stack in `band_names` order.

## Errors and faults

| Result | Trigger |
| --- | --- |
| `Err(FRAME_MALFORMED)` | Planes not 3-D, count mismatch, or unknown band name |

## Messages

None.

## Configuration

Uses `SensorConfig.mosaic_layout` and `InferenceConfig.input_bands`.

## Constraints

The module matches names only; it does not assume fixed band indices. `input_bands` must
be a subset of `mosaic_layout`.

## Related documents

- [`flight.payload.preprocess`](preprocess.md)
- [`flight.payload.preprocess.demosaic`](demosaic.md)
