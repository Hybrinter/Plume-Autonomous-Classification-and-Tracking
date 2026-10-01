"""Full-frame gate, blob overlap, and canvas scenes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from tools.ml_models.analysis.full_frame import (
    build_eval_scenes,
    score_dry_run,
    score_full_frame,
    score_tiled_frame,
    slice_tiles,
    stitch_tiles,
    summarize_frames,
    tile_hw_for_frame,
)
from tools.ml_models.data.canvas import CanvasConfig
from tools.ml_models.data.prism import FRAME_HW, TILE_GRID, TILE_HW


def test_empty_mask_with_a_blob_is_a_false_positive() -> None:
    """An empty ground truth, an open gate, and a blob is a false positive."""
    prob = np.zeros((20, 20), dtype=np.float32)
    prob[:4, :4] = 1.0
    gt = np.zeros((20, 20), dtype=np.float32)
    score = score_full_frame(1.0, prob, gt, placement="empty")
    assert score.empty_false_positive is True
    assert score.hit is False


def test_overlapping_blob_with_open_gate_is_a_hit() -> None:
    """A plume blob that overlaps the ground truth is a hit at logit 0."""
    gt = np.zeros((20, 20), dtype=np.float32)
    gt[8:12, 8:12] = 1.0
    score = score_full_frame(0.0, gt, gt, placement="center")
    assert score.hit is True
    assert score.empty_false_positive is False


def test_closed_gate_is_a_miss_on_a_perfect_mask() -> None:
    """A logit below 0 is a miss even when the mask matches the ground truth."""
    gt = np.zeros((20, 20), dtype=np.float32)
    gt[8:12, 8:12] = 1.0
    score = score_full_frame(-0.1, gt, gt)
    assert score.hit is False
    assert score.empty_false_positive is False


def _tiny_pack(root: Path) -> Path:
    """Write a 4-row pack whose test split has a background chip and a plume."""
    side = 8
    images = np.full((4, 3, side, side), 0.1, dtype=np.float32)
    masks = np.zeros((4, 1, side, side), dtype=np.float32)
    labels = np.zeros((4, 1), dtype=np.float32)
    masks[3, 0, 2:6, 2:6] = 1.0
    labels[3, 0] = 1.0
    images[3, :, 2:6, 2:6] = 0.9
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "images.npy", images)
    np.save(root / "masks.npy", masks)
    np.save(root / "labels.npy", labels)
    (root / "splits.json").write_text(
        json.dumps({"train": [0], "val": [1], "test": [2, 3]}) + "\n",
        encoding="utf-8",
    )
    return root


def test_build_eval_scenes_uses_a_tiny_frame(tmp_path: Path) -> None:
    """Scenes follow the caller canvas and stay far below 1544 by 2064."""
    pack = _tiny_pack(tmp_path / "pack")
    canvas = CanvasConfig(
        frame_hw=(24, 24),
        window_px=8,
        chip_side=8,
        feather_px=1,
        empty_fraction=0.0,
        max_plumes=1,
    )
    scenes = build_eval_scenes(pack, canvas, limit=1, seed=0)
    assert len(scenes) == 1
    assert scenes[0].image.shape == (3, 24, 24)
    assert scenes[0].mask.shape == (1, 24, 24)
    assert scenes[0].placement in {"center", "corner", "edge", "empty"}
    summary = summarize_frames((score_dry_run(scenes[0]),))
    assert "full_frame_hit_rate" in summary
    assert "chip_scores" in summary
    assert "tile_scores" in summary
    assert "chip_iou" in summary
    assert "chip_vs_frame_logit_margin" not in summary
    assert "passed" not in summary


def test_omitted_canvas_includes_plume_and_empty_scenes(tmp_path: Path) -> None:
    """An omitted canvas returns a plume scene and an empty scene."""
    pack = _tiny_pack(tmp_path / "pack")
    scenes = build_eval_scenes(pack, frame_h=24, frame_w=24, limit=1, seed=0)
    assert any(scene.label == 0.0 and scene.placement == "empty" for scene in scenes)
    assert any(scene.label > 0.0 for scene in scenes)
    summary = summarize_frames(tuple(score_dry_run(scene) for scene in scenes))
    assert summary["empty_frame_false_positive_rate"] is not None
    chip_scores = summary["chip_scores"]
    tile_scores = summary["tile_scores"]
    assert isinstance(chip_scores, dict)
    assert isinstance(tile_scores, dict)
    assert set(chip_scores) == {"logit", "iou"}
    assert set(tile_scores) == {"logit", "iou"}


def test_known_tile_lands_on_its_frame_pixels() -> None:
    """Tile (row, col) stitches back to that window."""
    tile_h, tile_w = 2, 3
    n_rows, n_cols = 4, 5
    frame = np.zeros((1, n_rows * tile_h, n_cols * tile_w), dtype=np.float32)
    row, col = 1, 2
    frame[:, row * tile_h : (row + 1) * tile_h, col * tile_w : (col + 1) * tile_w] = 0.7
    tiles = slice_tiles(frame, (tile_h, tile_w))
    assert tiles.shape == (n_rows * n_cols, 1, tile_h, tile_w)
    assert np.all(tiles[row * n_cols + col] == np.float32(0.7))
    assert float(tiles[0].sum()) == 0.0
    stitched = stitch_tiles(tiles, (n_rows * tile_h, n_cols * tile_w))
    assert np.array_equal(stitched, frame)


def test_flight_tile_roundtrip_keeps_one_window() -> None:
    """A 193 by 258 tile returns to its place on the 1544 by 2064 frame."""
    assert tile_hw_for_frame(FRAME_HW) == TILE_HW
    assert TILE_GRID[0] * TILE_HW[0] == FRAME_HW[0]
    assert TILE_GRID[1] * TILE_HW[1] == FRAME_HW[1]
    frame = np.zeros(FRAME_HW, dtype=np.float32)
    row, col = 3, 5
    tile_h, tile_w = TILE_HW
    frame[row * tile_h : (row + 1) * tile_h, col * tile_w : (col + 1) * tile_w] = 0.8
    tiles = slice_tiles(frame)
    assert tiles.shape == (TILE_GRID[0] * TILE_GRID[1], tile_h, tile_w)
    assert tiles[row * TILE_GRID[1] + col, 0, 0] == np.float32(0.8)
    stitched = stitch_tiles(tiles, FRAME_HW)
    assert stitched.shape == FRAME_HW
    assert np.array_equal(stitched, frame)


def test_score_tiled_frame_runs_one_batch_and_uses_the_max_logit() -> None:
    """One 64-tile batch feeds the gate, and the max logit opens it."""
    image = np.zeros((1, 24, 24), dtype=np.float32)
    gt = np.zeros((24, 24), dtype=np.float32)
    gt[:4, :4] = 1.0
    calls: list[tuple[int, ...]] = []

    def forward(tiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        calls.append(tuple(int(size) for size in tiles.shape))
        logits = np.full((tiles.shape[0],), -1.0, dtype=np.float64)
        logits[-1] = 0.0
        return logits, slice_tiles(gt)

    result = score_tiled_frame(image, gt, forward, placement="center")
    assert calls == [(64, 1, 3, 3)]
    assert result.tile_logit == 0.0
    assert result.score.hit is True
    assert np.array_equal(result.probability, gt)


def test_stitched_empty_blob_is_a_false_positive() -> None:
    """A 4 by 4 blob split across tiles is one false positive after the stitch."""
    image = np.zeros((1, 24, 24), dtype=np.float32)
    gt = np.zeros((24, 24), dtype=np.float32)
    prob = np.zeros_like(gt)
    prob[:4, :4] = 1.0

    def forward(tiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.ones((tiles.shape[0],), dtype=np.float64), slice_tiles(prob)

    result = score_tiled_frame(image, gt, forward, placement="empty")
    assert result.score.empty_false_positive is True
    assert result.score.hit is False
    assert np.array_equal(result.probability, prob)


def test_closed_max_tile_logit_is_a_miss() -> None:
    """A max tile logit below 0 misses even when the stitched mask matches."""
    image = np.zeros((1, 24, 24), dtype=np.float32)
    gt = np.zeros((24, 24), dtype=np.float32)
    gt[:4, :4] = 1.0

    def forward(tiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.full((tiles.shape[0],), -0.1, dtype=np.float64), slice_tiles(gt)

    result = score_tiled_frame(image, gt, forward)
    assert result.score.hit is False
    assert result.score.empty_false_positive is False


def test_frame_that_does_not_divide_into_eight_tiles_raises() -> None:
    """A frame that is not a multiple of the 8 by 8 grid is rejected."""
    frame = np.zeros((10, 16), dtype=np.float32)
    with pytest.raises(ValueError, match="tile grid"):
        slice_tiles(frame)
