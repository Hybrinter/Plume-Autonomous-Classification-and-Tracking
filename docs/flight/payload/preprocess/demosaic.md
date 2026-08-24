# flight.payload.preprocess.demosaic

**Source:** `packages/flight/src/flight/payload/preprocess/demosaic.py`
**Kind:** pure module

## Purpose

This module splits a 2x2 CFA mosaic into four half-resolution band planes. `interleave_bands`
rebuilds the mosaic from planes.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `CELL_OFFSETS` | constant | Row-major (row, col) offsets for the four CFA cells |
| `separate_bands` | function | Mosaic to `(4, H/2, W/2)` planes |
| `interleave_bands` | function | Planes to `(2*h, 2*w)` mosaic |

## Inputs and outputs

`separate_bands(mosaic)` returns `Result[np.ndarray, FaultCode]`.

`interleave_bands(planes)` returns `Result[np.ndarray, FaultCode]`.

## Behavior

1. `separate_bands` requires a 2-D array with even height and width.
2. Each plane k samples the mosaic at stride 2 from `CELL_OFFSETS[k]`.
3. Output stack shape is `(4, H/2, W/2)` float32.
4. `interleave_bands` scatters four `(h, w)` planes back into a `(2*h, 2*w)` mosaic.

Band names come from `SensorConfig.mosaic_layout` in the same cell order. This module is
layout-agnostic.

## Errors and faults

| Result | Trigger |
| --- | --- |
| `Err(FRAME_MALFORMED)` | Wrong rank, odd dimension, or plane count not four |

## Messages

None.

## Configuration

Mosaic size must match `SensorConfig.height_px` and `width_px`. The caller validates absolute
size; this module checks rank and parity only.

## Constraints

Pure module. No interpolation: planes are stride-2 extracts.

## Related documents

- [`flight.payload.preprocess`](preprocess.md)
- [`flight.payload.preprocess.band_select`](band_select.md)
