"""Full-frame gate, blob overlap, and canvas scenes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tools.ml_models.analysis.full_frame import (
    build_eval_scenes,
    score_dry_run,
    score_full_frame,
    summarize_frames,
)
from tools.ml_models.data.canvas import CanvasConfig


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
    assert "chip_iou" in summary
    assert "passed" not in summary


def test_omitted_canvas_includes_plume_and_empty_scenes(tmp_path: Path) -> None:
    """An omitted canvas returns a plume scene and an empty scene."""
    pack = _tiny_pack(tmp_path / "pack")
    scenes = build_eval_scenes(pack, frame_h=24, frame_w=24, limit=1, seed=0)
    assert any(scene.label == 0.0 and scene.placement == "empty" for scene in scenes)
    assert any(scene.label > 0.0 for scene in scenes)
    summary = summarize_frames(tuple(score_dry_run(scene) for scene in scenes))
    assert summary["empty_frame_false_positive_rate"] is not None
