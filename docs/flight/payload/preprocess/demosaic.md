# flight.payload.preprocess.demosaic

**Source:** `packages/flight/src/flight/payload/preprocess/demosaic.py`
**Kind:** pure module

## Purpose

This module splits a 2x2 CFA mosaic into four registered band planes and can rebuild a
mosaic from planes. Separation uses stride-2 sampling with no interpolation.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `CELL_OFFSETS` | constant | Row-major (row, col) offsets for each 2x2 cell |
| `separate_bands` | function | Splits `(H, W)` mosaic into `(4, H/2, W/2)` planes |
| `interleave_bands` | function | Rebuilds `(2*h, 2*w)` mosaic from `(4, h, w)` planes |

## Inputs and outputs

`separate_bands(mosaic)` returns `Result[np.ndarray, FaultCode]` with shape
`(4, H/2, W/2)` float32.

`interleave_bands(planes)` returns `Result[np.ndarray, FaultCode]` with shape
`(2*h, 2*w)`.

## Behavior

1. `separate_bands` checks that the input is 2-D with even height and width.
2. For each entry in `CELL_OFFSETS`, it samples `mosaic[r::2, c::2]` and stacks four
   planes.
3. `interleave_bands` checks for exactly four planes, allocates a mosaic, and scatters
   each plane back into its cell offset.

Band names come from `SensorConfig.mosaic_layout`; this module is layout-agnostic.

## Errors and faults

| Result | Trigger |
| --- | --- |
| `Err(FRAME_MALFORMED)` | Mosaic not 2-D, odd dimension, planes not rank-3, or plane count not four |

## Messages

None.

## Configuration

None. The caller must supply a mosaic whose size matches `SensorConfig`.

## Constraints

Planes are half the mosaic resolution and spatially registered. `interleave_bands` is
the exact inverse of `separate_bands` and is used by the sim scene renderer.

## Related documents

- [`flight.payload.preprocess`](preprocess.md)
- [`flight.payload.preprocess.band_select`](band_select.md)
