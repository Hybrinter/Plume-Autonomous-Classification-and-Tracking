"""Dataset adapters for flight frame and tile geometry.

The flight inference configuration is the source of frame, grid, band, and
reference-GSD defaults. Tiling work is delegated to the flight preprocessing
contract; this module keeps the established tools API and exceptions.
"""

from __future__ import annotations

import numpy as np
from flight.libs.config import InferenceConfig
from flight.libs.types import Err
from flight.payload.preprocess.tiling import slice_frame as _flight_slice_frame
from flight.payload.preprocess.tiling import stitch_tiles as _flight_stitch_tiles

_DEFAULT = InferenceConfig()
FRAME_H_PX = _DEFAULT.input_height_px
FRAME_W_PX = _DEFAULT.input_width_px
GRID_ROWS = _DEFAULT.tile_rows
GRID_COLS = _DEFAULT.tile_cols
TILE_H_PX = FRAME_H_PX // GRID_ROWS
TILE_W_PX = FRAME_W_PX // GRID_COLS
GSD_REFERENCE_M = _DEFAULT.gsd_reference_m
INPUT_BANDS = _DEFAULT.input_bands


def frame_hw() -> tuple[int, int]:
    """Return the configured flight frame size as ``(height, width)``."""
    return (FRAME_H_PX, FRAME_W_PX)


def grid_hw() -> tuple[int, int]:
    """Return the configured tile grid as ``(rows, cols)``."""
    return (GRID_ROWS, GRID_COLS)


def tile_hw() -> tuple[int, int]:
    """Return one configured flight tile as ``(height, width)``."""
    return (TILE_H_PX, TILE_W_PX)


def slice_frame(frame: np.ndarray) -> np.ndarray:
    """Slice one flight frame using flight tiling and preserve the tools API.

    Raises:
        ValueError: If the frame does not match the configured flight layout.
    """
    result = _flight_slice_frame(frame, grid_hw())
    if isinstance(result, Err):
        raise ValueError(f"flight frame tiling rejected frame shape {frame.shape}: {result.error}")
    if frame.shape[-2:] != frame_hw():
        raise ValueError(
            f"flight frame layout must be {FRAME_H_PX}x{FRAME_W_PX}; "
            f"got {frame.shape[-2]}x{frame.shape[-1]}"
        )
    return result.value


def stitch_tiles(tiles: np.ndarray) -> np.ndarray:
    """Stitch flight tiles and preserve the tools single-channel squeeze.

    Raises:
        ValueError: If tiles do not match the configured flight layout.
    """
    result = _flight_stitch_tiles(tiles, grid_hw())
    if isinstance(result, Err):
        raise ValueError(f"flight tile stitching rejected shape {tiles.shape}: {result.error}")
    frame = result.value
    expected_hw = (TILE_H_PX * GRID_ROWS, TILE_W_PX * GRID_COLS)
    if frame.shape[-2:] != expected_hw:
        raise ValueError(f"flight tile stitching returned unexpected shape {frame.shape}")
    if frame.shape[0] == 1:
        return np.asarray(frame[0])
    return frame
