"""Canvas train loop on a tiny flight frame."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import torch
from tools.ml_models.data.canvas import CanvasConfig
from tools.ml_models.data.meta import Provenance
from tools.ml_models.data.pack import write_processed_pack
from tools.ml_models.data.split import SplitIndex
from tools.ml_models.train.config import TrainConfig, load_train_config
from tools.ml_models.train.loop import train


def _write_pack(dest: Path) -> None:
    """Write four 8x8 chips: negatives, one classifier-only row, one polygon."""
    images = np.zeros((4, 3, 8, 8), dtype=np.float32)
    masks = np.zeros((4, 1, 8, 8), dtype=np.float32)
    labels = np.zeros((4, 1), dtype=np.float32)
    images[0] = 0.1
    images[1] = 0.2
    images[2] = 0.05
    images[2, :, 2:6, 2:6] = 1.0
    images[3] = 0.3
    masks[2, 0, 2:6, 2:6] = 1.0
    labels[1, 0] = 1.0
    labels[2, 0] = 1.0
    provenance = Provenance(
        ingest_path="flight_camera",
        radiometry="normalize_dn",
        gsd_m=10.0,
        extent_m=20.0,
        weight_table_id="table-a",
        band_names=("b0", "b1", "b2"),
        norm="unit",
        bit_depth=12,
    )
    write_processed_pack(
        dest,
        images,
        masks,
        labels,
        provenance,
        "",
        splits=SplitIndex(train=(0, 1, 2), val=(3,), test=()),
    )


def _canvas() -> CanvasConfig:
    """Return the tiny frame used by the loop tests."""
    return CanvasConfig(
        frame_hw=(20, 24),
        window_px=12,
        full_frame_every=2,
        chip_side=8,
        empty_fraction=0.0,
        max_plumes=1,
        feather_px=1,
        seed=0,
    )


def _config(
    pack: Path,
    tmp_path: Path,
    *,
    kind: Literal["classifier", "segmentor"],
    run_id: str,
    channels: int = 3,
) -> TrainConfig:
    """Return a two-step canvas config."""
    return TrainConfig(
        kind=kind,
        epochs=1,
        batch_size=2,
        max_steps=2,
        in_channels=channels,
        data_dir=str(pack),
        run_dir=str(tmp_path / "runs"),
        run_id=run_id,
        seed=0,
        device="cpu",
        canvas=_canvas(),
    )


def test_canvas_loop_writes_frame_checkpoint(tmp_path: Path) -> None:
    """Two steps see one full frame and one window, and best.pt stores both sizes."""
    pack = tmp_path / "pack"
    _write_pack(pack)
    root = train(_config(pack, tmp_path, kind="segmentor", run_id="canvas"))
    assert root.is_dir()
    assert (root / "history.csv").is_file()
    assert (root / "summary.json").is_file()
    shapes = json.loads((root / "batch_shapes.json").read_text(encoding="utf-8"))
    assert shapes == [[2, 3, 20, 24], [2, 3, 12, 12]]
    payload = torch.load(root / "checkpoints" / "best.pt", map_location="cpu", weights_only=True)
    assert tuple(payload["frame_hw"]) == (20, 24)
    assert payload["window_px"] == 12
    assert payload["in_channels"] == 3
    assert payload["arch"] == "dilatenet"
    assert payload["band_names"] == ["b0", "b1", "b2"]
    assert payload["dataset_hash"]
    assert "state_dict" in payload
    assert payload["epoch"] == 1
    last = torch.load(root / "checkpoints" / "last.pt", map_location="cpu", weights_only=True)
    assert tuple(last["frame_hw"]) == (20, 24)
    assert last["window_px"] == 12
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert summary["dataset_hash"] == payload["dataset_hash"]
    assert summary["val_metric"] == "mean_dice"
    assert summary["best_val_metric"] is not None
    assert "test_mean_dice" not in summary
    assert summary["n_test"] == 0
    loaded = load_train_config(str(root / "config.toml"))
    assert loaded.canvas is not None
    assert loaded.canvas.frame_hw == (20, 24)
    assert loaded.canvas.window_px == 12
    assert loaded.max_steps == 2


def test_canvas_classifier_runs_two_steps(tmp_path: Path) -> None:
    """A classifier canvas run selects the checkpoint with full-frame F1."""
    pack = tmp_path / "pack"
    _write_pack(pack)
    root = train(_config(pack, tmp_path, kind="classifier", run_id="canvas-cls"))
    shapes = json.loads((root / "batch_shapes.json").read_text(encoding="utf-8"))
    assert shapes == [[2, 3, 20, 24], [2, 3, 12, 12]]
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert summary["val_metric"] == "f1"
    assert summary["loss"] == "bce+focal_dice"
    assert summary["arch"] == "pactnet"
    payload = torch.load(root / "checkpoints" / "best.pt", map_location="cpu", weights_only=True)
    assert tuple(payload["frame_hw"]) == (20, 24)
    assert payload["window_px"] == 12


def test_canvas_channel_mismatch_raises(tmp_path: Path) -> None:
    """A pack channel count that differs from in_channels raises."""
    pack = tmp_path / "pack"
    _write_pack(pack)
    config = _config(pack, tmp_path, kind="segmentor", run_id="bad", channels=4)
    with pytest.raises(ValueError, match="in_channels=4"):
        train(config)
