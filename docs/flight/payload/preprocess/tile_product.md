# flight.payload.preprocess.tile_product

**Source:** `packages/flight/src/flight/payload/preprocess/tile_product.py`  
**Kind:** pure module

## Purpose

This module defines the schema-2 unit-tile contract shared by flight
processing and the ground importer. A layout records the bands, tile size,
grid, and GSD reference of one tile product; a capture records one tile's
provenance. Pure validators check layouts, captures, and unit-tile images.

Satisfies: REQ-AIML-PREP-002.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `UNIT_TILE_SOURCE_SCHEMA` | constant | On-disk schema identifier, 2 |
| `UnitTileLayout` | dataclass | Frozen `(band_names, tile_hw, grid, gsd_reference_m)` record |
| `UnitTileCapture` | dataclass | Frozen `(tile_id, frame_id, grid_rc, theta_g_deg, gsd, gsd_nominal)` record |
| `validate_layout` | function | Validate a recorded layout |
| `validate_capture` | function | Validate one capture against a layout |
| `validate_unit_tile` | function | Validate one unit-domain image against a layout |

## Inputs and outputs

`validate_layout(layout)` accepts a `UnitTileLayout` and returns
`Ok(None)` or `Err(FRAME_MALFORMED)`. `band_names` must be a nonempty
tuple of nonempty unique strings in storage order. `tile_hw` and `grid`
must be two-element tuples of positive Python integers. `gsd_reference_m`
must be a finite real number greater than 0.

`validate_capture(capture, layout)` returns `Err(FRAME_MALFORMED)` when the
layout itself is invalid, then checks the capture: `tile_id` is a nonempty
file stem without `.`, `..`, separators, or NUL; `frame_id` is nonempty;
`grid_rc` is a `(row, col)` pair of nonnegative integers inside the layout
grid; `theta_g_deg` is a finite real number with no imposed angle bounds;
`gsd` is a `GsdPair` with finite positive components; `gsd_nominal` is a
real boolean.

`validate_unit_tile(image, layout)` returns `Ok(None)` when `image` is a
float32 NumPy array of shape `(len(band_names), height, width)` with all
pixels finite inside `[0, 1]`.

## Behavior

1. Each validator checks the runtime type of every field before reading it,
   and rejects a non-`UnitTileLayout` or invalid layout passed to the other
   validators.
2. Captures carry no label or ground-truth mask; labels and reviewed masks
   remain separate ground-side records.
3. Validation performs no casting, clipping, or mutation of the image or
   metadata.

## Errors and faults

All invalid inputs return `Err(FaultCode.FRAME_MALFORMED)`. The validators
never raise, log, read clocks, or touch files.

## Messages

None. This is a pure module.

## Configuration

None. The ground importer records the layout in `source.json`.

## Constraints

Imports are limited to NumPy, the standard library, `flight.libs.types`,
and `flight.payload.gimbal.footprint`. No Torch, hardware SDK, bus access,
or tools dependency is permitted.

## Related documents

- [`flight.payload.preprocess`](../preprocess.md)
- [`flight.payload.preprocess.tiling`](tiling.md)
- [`flight.payload.gimbal.footprint`](../../gimbal/footprint.md)
