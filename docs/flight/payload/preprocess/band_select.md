# flight.payload.preprocess.band_select

**Source:** `packages/flight/src/flight/payload/preprocess/band_select.py`
**Kind:** pure module

## Purpose

This module reorders demosaicked band planes from `mosaic_layout` order into the model
`input_bands` order.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `select_bands` | function | Gathers planes by band name |

## Inputs and outputs

`select_bands(planes, layout, band_names)` returns `Result[np.ndarray, FaultCode]`.

Input planes shape is `(len(layout), H, W)`. Output shape is `(len(band_names), H, W)`.

## Behavior

1. Verify planes are 3-D and the plane count matches `layout` length.
2. Resolve each name in `band_names` to an index in `layout`.
3. Return a fancy-indexed stack in `band_names` order.

Unknown names or count mismatch return `Err(FRAME_MALFORMED)`.

## Errors and faults

| Result | Trigger |
| --- | --- |
| `Err(FRAME_MALFORMED)` | Rank error, plane count mismatch, unknown band name |

## Messages

None.

## Configuration

Uses `SensorConfig.mosaic_layout` and `InferenceConfig.input_bands`.

## Constraints

Pure module. Layout-agnostic name matching only; no fixed band index table.

## Related documents

- [`flight.payload.preprocess`](preprocess.md)
- [`flight.payload.preprocess.demosaic`](demosaic.md)
