"""Tests for the local flight tile geometry."""

import numpy as np
import pytest
from flight.libs.config import InferenceConfig
from flight.libs.types import Ok
from tools.ml_models.dataset import geometry
from tools.ml_models.dataset.geometry import (
    FRAME_H_PX,
    FRAME_W_PX,
    slice_frame,
    stitch_tiles,
    tile_hw,
)


def test_slice_and_stitch_round_trip() -> None:
    """Slicing a frame and stitching the tiles restores the frame."""
    frame = np.arange(FRAME_H_PX * FRAME_W_PX, dtype=np.float32).reshape(
        1, 1, FRAME_H_PX, FRAME_W_PX
    )
    tiles = slice_frame(frame)
    height, width = tile_hw()
    assert tiles.shape == (64, 1, height, width)
    restored = stitch_tiles(tiles)
    np.testing.assert_array_equal(restored, frame[0, 0])


def test_defaults_derive_from_flight_inference_config() -> None:
    """Dataset aliases follow the flight-owned default geometry and bands."""
    config = InferenceConfig()
    assert geometry.frame_hw() == (config.input_height_px, config.input_width_px)
    assert geometry.grid_hw() == (config.tile_rows, config.tile_cols)
    assert geometry.tile_hw() == (
        config.input_height_px // config.tile_rows,
        config.input_width_px // config.tile_cols,
    )
    assert geometry.INPUT_BANDS == config.input_bands
    assert geometry.GSD_REFERENCE_M == config.gsd_reference_m


def test_wrappers_delegate_layout_and_preserve_flight_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dataset functions call flight tiling with its configured grid."""
    calls: list[tuple[str, tuple[int, int]]] = []

    def fake_slice(frame: np.ndarray, grid: tuple[int, int]) -> Ok[np.ndarray]:
        calls.append(("slice", grid))
        return Ok(np.zeros((grid[0] * grid[1], frame.shape[1], *geometry.tile_hw())))

    def fake_stitch(tiles: np.ndarray, grid: tuple[int, int]) -> Ok[np.ndarray]:
        calls.append(("stitch", grid))
        return Ok(np.zeros((1, *geometry.frame_hw())))

    monkeypatch.setattr(geometry, "_flight_slice_frame", fake_slice)
    monkeypatch.setattr(geometry, "_flight_stitch_tiles", fake_stitch)
    frame = np.zeros((1, 1, *geometry.frame_hw()))
    assert slice_frame(frame).shape == (64, 1, *geometry.tile_hw())
    assert stitch_tiles(np.zeros((64, 1, *geometry.tile_hw()))).shape == geometry.frame_hw()
    assert calls == [("slice", geometry.grid_hw()), ("stitch", geometry.grid_hw())]


def test_flight_tiling_errors_become_value_errors() -> None:
    """Malformed layouts keep the tools-facing ValueError contract."""
    with pytest.raises(ValueError, match="flight frame layout"):
        slice_frame(np.zeros((1, 1, 8, 8)))
    with pytest.raises(ValueError, match="flight tile stitching rejected"):
        stitch_tiles(np.zeros((63, 1, *tile_hw())))
