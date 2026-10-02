# flight.payload.preprocess.tiling

**Source:** `packages/flight/src/flight/payload/preprocess/tiling.py`  
**Kind:** pure module

## Purpose

This module divides full-sensor frames into equal row-major tiles and stitches tiled outputs back into a channel-major frame. It preserves channel axes, including a singleton channel.

Satisfies: REQ-AIML-PREP-002.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `validate_layout` | function | Validate frame and grid dimensions and return tile size |
| `slice_frame` | function | Split a single batched frame into row-major tiles |
| `stitch_tiles` | function | Stitch row-major tiles into a channel-major frame |

## Inputs and outputs

`validate_layout(frame_hw, grid)` takes `(height, width)` and `(rows, cols)` integer tuples. It returns tile `(height, width)` in `Ok`, or `Err(FRAME_MALFORMED)` if dimensions are invalid or the grid does not divide the frame.

`slice_frame(frame, grid)` takes a `(1, channels, height, width)` NumPy array and returns `(rows * cols, channels, tile_height, tile_width)` tiles in row-major order. The default grid is 8 × 8.

`stitch_tiles(tiles, grid)` takes `(rows * cols, channels, tile_height, tile_width)` and returns `(channels, height, width)`. It always preserves the channel axis.

## Behavior

1. Validate positive frame dimensions and a positive integer grid.
2. Require the grid to divide the frame dimensions exactly.
3. Copy each tile in row-major order, with tile index `row * cols + col`.
4. Stitch by placing each row-major tile into its matching frame region.

## Errors and faults

`validate_layout`, `slice_frame`, and `stitch_tiles` return `Err(FRAME_MALFORMED)` for malformed dimensions, arrays, channel counts, tile counts, or grids.

## Messages

None. This is a pure module.

## Configuration

The caller supplies the grid. Flight defaults are `InferenceConfig.tile_rows = 8` and `InferenceConfig.tile_cols = 8`. The default 1544 × 2064 sensor frame produces 193 × 258 tiles; configured input dimensions continue to describe the full sensor frame.

## Constraints

The input to `slice_frame` must contain exactly one frame. Grids must divide both frame dimensions without remainder. Tiling performs no resizing, normalization, or I/O.

## Related documents

- [`flight.payload.preprocess`](../preprocess.md)
- [`flight.libs.config.config`](../../libs/config/config.md)
