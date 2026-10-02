# tools.ml_models.dataset.geometry

**Source:** `packages/tools/src/tools/ml_models/dataset/geometry.py`

**Kind:** module

## Purpose

This module adapts the flight-owned inference geometry and tiling contract to
the established dataset tools API. Default frame dimensions, grid, tile size,
GSD reference, and input bands come from `InferenceConfig`.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `FRAME_H_PX` / `FRAME_W_PX` | constant | Configured flight frame dimensions |
| `GRID_ROWS` / `GRID_COLS` | constant | Configured flight tile grid |
| `TILE_H_PX` / `TILE_W_PX` | constant | Derived flight tile dimensions |
| `GSD_REFERENCE_M` | constant | Configured flight reference GSD |
| `INPUT_BANDS` | constant | Configured flight channel order |
| `frame_hw` / `grid_hw` / `tile_hw` | function | Return the flight sizes |
| `slice_frame` | function | Delegate slicing to flight tiling |
| `stitch_tiles` | function | Delegate stitching and squeeze one channel |

## Inputs and outputs

`slice_frame(frame) -> np.ndarray` takes `(1, C, H, W)` and returns row-major
tiles `(rows * cols, C, tile_h, tile_w)`. `stitch_tiles(tiles) -> np.ndarray`
returns `(C, H, W)`, or `(H, W)` for a single channel.

## Behavior

1. Flight `InferenceConfig` supplies the default frame dimensions, grid,
   reference GSD, and band order.
2. Flight preprocessing validates and performs the tile copy operations.
3. Flight `Result` errors become `ValueError` for compatibility with callers
   of the tools API.

## Errors and faults

`ValueError` when flight tiling rejects an input or returns an unexpected
layout.

## Messages

None.

## Configuration

The public module constants are aliases derived from a default
`InferenceConfig`. The dataset build specification can still override its
input bands and GSD reference for a specific dataset.

## Constraints

This module does not import torch. Flight remains the source of live tiling
behavior.

## Related documents

- [`tools.ml_models.dataset`](../dataset.md)
- [`tools.ml_models.dataset.build`](build.md)
- [`flight.payload.preprocess.tiling`](../../../../flight/payload/preprocess/tiling.md)
