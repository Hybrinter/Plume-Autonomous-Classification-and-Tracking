"""Pure row-major tiling for full-sensor flight frames.

Frames remain full-resolution until this module splits them into equal tiles.
The leading batch axis is exactly one frame, and channel axes are preserved.

Satisfies: REQ-AIML-PREP-002.
"""

from __future__ import annotations

import numpy as np

from flight.libs.types import Err, FaultCode, Ok, Result


def validate_layout(
    frame_hw: tuple[int, int], grid: tuple[int, int]
) -> Result[tuple[int, int], FaultCode]:
    """Return tile ``(height, width)`` if a positive grid divides a frame."""
    if (
        not isinstance(frame_hw, tuple)
        or len(frame_hw) != 2
        or not isinstance(grid, tuple)
        or len(grid) != 2
        or any(
            not isinstance(value, int) or isinstance(value, bool) for value in (*frame_hw, *grid)
        )
    ):
        return Err(FaultCode.FRAME_MALFORMED)
    height, width = frame_hw
    rows, cols = grid
    if min(height, width, rows, cols) <= 0 or height % rows or width % cols:
        return Err(FaultCode.FRAME_MALFORMED)
    return Ok((height // rows, width // cols))


def slice_frame(frame: np.ndarray, grid: tuple[int, int] = (8, 8)) -> Result[np.ndarray, FaultCode]:
    """Convert a ``(1, C, H, W)`` frame to row-major ``(R*C, C, h, w)`` tiles."""
    if not isinstance(frame, np.ndarray) or frame.ndim != 4 or frame.shape[0] != 1:
        return Err(FaultCode.FRAME_MALFORMED)
    _, channels, height, width = frame.shape
    if channels < 1:
        return Err(FaultCode.FRAME_MALFORMED)
    layout = validate_layout((height, width), grid)
    if isinstance(layout, Err):
        return layout
    tile_h, tile_w = layout.value
    rows, cols = grid
    tiles = np.empty((rows * cols, channels, tile_h, tile_w), dtype=frame.dtype)
    index = 0
    for row in range(rows):
        for col in range(cols):
            tiles[index] = frame[
                0,
                :,
                row * tile_h : (row + 1) * tile_h,
                col * tile_w : (col + 1) * tile_w,
            ]
            index += 1
    return Ok(tiles)


def stitch_tiles(
    tiles: np.ndarray, grid: tuple[int, int] = (8, 8)
) -> Result[np.ndarray, FaultCode]:
    """Convert ``(R*C, channels, h, w)`` tiles to ``(channels, H, W)``."""
    if not isinstance(tiles, np.ndarray) or tiles.ndim != 4:
        return Err(FaultCode.FRAME_MALFORMED)
    count, channels, tile_h, tile_w = tiles.shape
    if tile_h < 1 or tile_w < 1 or channels < 1:
        return Err(FaultCode.FRAME_MALFORMED)
    if (
        not isinstance(grid, tuple)
        or len(grid) != 2
        or any(not isinstance(value, int) or isinstance(value, bool) for value in grid)
        or min(grid) <= 0
    ):
        return Err(FaultCode.FRAME_MALFORMED)
    rows, cols = grid
    if count != rows * cols:
        return Err(FaultCode.FRAME_MALFORMED)
    frame_h, frame_w = rows * tile_h, cols * tile_w
    frame = np.empty((channels, frame_h, frame_w), dtype=tiles.dtype)
    index = 0
    for row in range(rows):
        for col in range(cols):
            frame[
                :,
                row * tile_h : (row + 1) * tile_h,
                col * tile_w : (col + 1) * tile_w,
            ] = tiles[index]
            index += 1
    return Ok(frame)
