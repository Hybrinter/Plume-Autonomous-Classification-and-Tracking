# tools.ml_models.dataset.geometry

**Source:** `packages/tools/src/tools/ml_models/dataset/geometry.py`
**Kind:** module

## Purpose

This module holds the flight frame, grid, and tile geometry constants used
while a dataset is built, plus frame-to-tile slicing helpers.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `FRAME_H_PX` / `FRAME_W_PX` | constant | Flight frame 1544 by 2064 |
| `GRID_ROWS` / `GRID_COLS` | constant | Tile grid 8 by 8 |
| `TILE_H_PX` / `TILE_W_PX` | constant | Flight tile 193 by 258 |
| `GSD_REFERENCE_M` | constant | Reference GSD of 15.87 metres, re-exported from `flight.payload.gimbal.footprint` |
| `INPUT_BANDS` | constant | `BLUE`, `GREEN`, `RED` channel order |
| `frame_hw` / `grid_hw` / `tile_hw` | function | Return the sizes above as tuples |
| `slice_frame` | function | Cut one frame into 64 tiles |
| `stitch_tiles` | function | Paste 64 tiles back into a frame |

## Inputs and outputs

`slice_frame(frame) -> np.ndarray` takes `(1, C, 1544, 2064)` and returns
`(64, C, 193, 258)` in row-major order. Index `row * 8 + col` selects the
tile at grid `(row, col)`.

`stitch_tiles(tiles) -> np.ndarray` takes `(64, C, 193, 258)` and returns
`(1544, 2064)` when `C` is 1, otherwise `(C, 1544, 2064)`.

## Behavior

1. H is along-track and W is lateral.
2. `slice_frame` copies each tile out of the single-frame batch.
3. `stitch_tiles` writes each tile into its grid position and drops the
   channel axis when `C` is 1.

## Errors and faults

`ValueError` when `slice_frame` does not receive a single 1544 by 2064
frame, or when `stitch_tiles` does not receive 64 tiles of 193 by 258.

## Messages

None.

## Configuration

The constants are fixed at module level. There is no TOML file.

## Constraints

`GSD_REFERENCE_M` comes from `flight.payload.gimbal.footprint` so the
dataset build and inference share one reference. The frame and grid
constants stay local to this module until flight tiling publishes the
same numbers. This module does not import torch.

## Related documents

- [`tools.ml_models.dataset`](../dataset.md)
- [`tools.ml_models.dataset.build`](build.md)
- [`tools.ml_models.dataset.sources.flight`](sources/flight.md)
- [`flight.payload.gimbal.footprint`](../../../../flight/payload/gimbal/footprint.md)
