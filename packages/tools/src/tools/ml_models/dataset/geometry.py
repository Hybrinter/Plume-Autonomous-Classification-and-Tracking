"""Flight frame, grid, and tile geometry used while a dataset is built.

Contains:
  - FRAME_H_PX, FRAME_W_PX, GRID_ROWS, GRID_COLS, TILE_H_PX, TILE_W_PX.
  - GSD_REFERENCE_M, INPUT_BANDS.
  - frame_hw, grid_hw, tile_hw.
  - slice_frame, stitch_tiles.

H is along-track and W is lateral. The 1544 by 2064 frame divides into an
8 by 8 grid of 193 by 258 tiles. ``GSD_REFERENCE_M`` is re-exported from
``flight.payload.gimbal.footprint`` so build and inference share one
reference. The frame and grid constants stay local until flight tiling is
the source of the same numbers.
"""

from __future__ import annotations

import numpy as np
from flight.payload.gimbal.footprint import GSD_REFERENCE_M

__all__ = [
    "FRAME_H_PX",
    "FRAME_W_PX",
    "GRID_COLS",
    "GRID_ROWS",
    "GSD_REFERENCE_M",
    "INPUT_BANDS",
    "TILE_H_PX",
    "TILE_W_PX",
    "frame_hw",
    "grid_hw",
    "slice_frame",
    "stitch_tiles",
    "tile_hw",
]

FRAME_H_PX = 1544
FRAME_W_PX = 2064
GRID_ROWS = 8
GRID_COLS = 8
TILE_H_PX = FRAME_H_PX // GRID_ROWS
TILE_W_PX = FRAME_W_PX // GRID_COLS
INPUT_BANDS: tuple[str, ...] = ("BLUE", "GREEN", "RED")


def frame_hw() -> tuple[int, int]:
    """Return the flight frame size as ``(height, width)``.

    Returns:
        tuple[int, int]: ``(1544, 2064)``.
    """
    return (FRAME_H_PX, FRAME_W_PX)


def grid_hw() -> tuple[int, int]:
    """Return the tile grid as ``(rows, cols)``.

    Returns:
        tuple[int, int]: ``(8, 8)``.
    """
    return (GRID_ROWS, GRID_COLS)


def tile_hw() -> tuple[int, int]:
    """Return one flight tile as ``(height, width)``.

    Returns:
        tuple[int, int]: ``(193, 258)``.
    """
    return (TILE_H_PX, TILE_W_PX)


def slice_frame(frame: np.ndarray) -> np.ndarray:
    """Cut a flight frame into 64 tiles in row-major order.

    Args:
        frame: np.ndarray[(1, C, 1544, 2064)] batch of one frame.

    Returns:
        np.ndarray[(64, C, 193, 258)]: Tiles. Index ``row * 8 + col``.

    Raises:
        ValueError: If the array is not a single 1544 by 2064 frame.
    """
    if frame.ndim != 4 or frame.shape[0] != 1:
        raise ValueError(f"frame must have shape (1, C, H, W); got {frame.shape}")
    _batch, channels, height, width = frame.shape
    if (height, width) != (FRAME_H_PX, FRAME_W_PX):
        raise ValueError(f"frame must be {FRAME_H_PX}x{FRAME_W_PX}; got {height}x{width}")
    tiles = np.empty((GRID_ROWS * GRID_COLS, channels, TILE_H_PX, TILE_W_PX), dtype=frame.dtype)
    index = 0
    for row in range(GRID_ROWS):
        row_slice = slice(row * TILE_H_PX, (row + 1) * TILE_H_PX)
        for col in range(GRID_COLS):
            col_slice = slice(col * TILE_W_PX, (col + 1) * TILE_W_PX)
            tiles[index] = frame[0, :, row_slice, col_slice]
            index += 1
    return tiles


def stitch_tiles(tiles: np.ndarray) -> np.ndarray:
    """Paste 64 tiles back into a flight frame.

    Args:
        tiles: np.ndarray[(64, C, 193, 258)] in row-major order.

    Returns:
        np.ndarray[(1544, 2064)] when ``C`` is 1, otherwise
        np.ndarray[(C, 1544, 2064)].

    Raises:
        ValueError: If the leading dimension is not 64 or the tile size differs
            from 193 by 258.
    """
    if tiles.ndim != 4 or tiles.shape[0] != GRID_ROWS * GRID_COLS:
        raise ValueError(f"tiles must have shape (64, C, H, W); got {tiles.shape}")
    _count, channels, height, width = tiles.shape
    if (height, width) != (TILE_H_PX, TILE_W_PX):
        raise ValueError(f"tile must be {TILE_H_PX}x{TILE_W_PX}; got {height}x{width}")
    frame = np.empty((channels, FRAME_H_PX, FRAME_W_PX), dtype=tiles.dtype)
    index = 0
    for row in range(GRID_ROWS):
        row_slice = slice(row * TILE_H_PX, (row + 1) * TILE_H_PX)
        for col in range(GRID_COLS):
            col_slice = slice(col * TILE_W_PX, (col + 1) * TILE_W_PX)
            frame[:, row_slice, col_slice] = tiles[index]
            index += 1
    if channels == 1:
        return np.asarray(frame[0])
    return frame
