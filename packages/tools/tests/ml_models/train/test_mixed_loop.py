"""Mixed-extent train loop on a chip pack and a tile pack."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import torch
from tools.ml_models.data.meta import Provenance
from tools.ml_models.data.pack import load_processed_pack, write_processed_pack
from tools.ml_models.data.prism import union_location_split
from tools.ml_models.data.split import SplitRecipe
from tools.ml_models.train.config import TrainConfig, load_train_config
from tools.ml_models.train.loop import train
from tools.ml_models.train.losses import bce_per_sample, focal_per_sample, weighted_batch_loss

_GROUPS = ("site-a", "site-b", "site-c", "site-d", "site-a")


def _provenance() -> Provenance:
    """Return one provenance shared by the chip pack and the tile pack."""
    return Provenance(
        ingest_path="flight_camera",
        radiometry="normalize_dn",
        gsd_m=10.0,
        extent_m=20.0,
        weight_table_id="table-a",
        band_names=("b0", "b1", "b2"),
        norm="unit",
        bit_depth=12,
    )


def _write_pack(dest: Path, hw: tuple[int, int]) -> None:
    """Write five rows. The on-disk split uses recipe seed 1."""
    height, width = hw
    images = np.zeros((5, 3, height, width), dtype=np.float32)
    masks = np.zeros((5, 1, height, width), dtype=np.float32)
    labels = np.zeros((5, 1), dtype=np.float32)
    images[0] = 0.1
    images[1] = 0.2
    images[2] = 0.05
    images[2, :, 2:6, 2:6] = 1.0
    images[3] = 0.3
    images[4] = 0.15
    masks[2, 0, 2:6, 2:6] = 1.0
    labels[1, 0] = 1.0
    labels[2, 0] = 1.0
    write_processed_pack(
        dest,
        images,
        masks,
        labels,
        _provenance(),
        "",
        group_ids=_GROUPS,
        recipe=SplitRecipe(seed=1),
    )


def _config(
    chip: Path,
    tile: Path,
    tmp_path: Path,
    *,
    kind: Literal["classifier", "segmentor"],
    run_id: str,
    channels: int = 3,
) -> TrainConfig:
    """Return a two-step mixed-extent config."""
    return TrainConfig(
        kind=kind,
        epochs=1,
        batch_size=2,
        max_steps=2,
        in_channels=channels,
        chip_dir=str(chip),
        tile_dir=str(tile),
        chip_weight=1.5,
        tile_weight=0.5,
        run_dir=str(tmp_path / "runs"),
        run_id=run_id,
        seed=0,
        device="cpu",
    )


def test_bce_and_focal_match_across_spatial_size() -> None:
    """Per-image BCE and focal stay equal when the map grows."""
    logit = 0.3
    small_logits = torch.full((2, 1, 8, 8), logit)
    large_logits = torch.full((2, 1, 24, 32), logit)
    small_targets = torch.ones((2, 1, 8, 8))
    large_targets = torch.ones((2, 1, 24, 32))
    small_bce = bce_per_sample(small_logits, small_targets)
    large_bce = bce_per_sample(large_logits, large_targets)
    assert torch.allclose(small_bce, large_bce)
    small_focal = focal_per_sample(small_logits, small_targets, gamma=2.0, alpha=0.25)
    large_focal = focal_per_sample(large_logits, large_targets, gamma=2.0, alpha=0.25)
    assert torch.allclose(small_focal, large_focal)
    assert torch.allclose(
        weighted_batch_loss(large_bce, 0.5),
        small_bce.mean() * 0.5,
    )


def test_mixed_loop_writes_tile_checkpoint(tmp_path: Path) -> None:
    """Two steps are one chip batch and one tile batch. best.pt stores both sizes."""
    chip = tmp_path / "chips"
    tile = tmp_path / "tiles"
    _write_pack(chip, (16, 16))
    _write_pack(tile, (20, 24))
    root = train(_config(chip, tile, tmp_path, kind="segmentor", run_id="mixed"))
    assert root.is_dir()
    assert (root / "history.csv").is_file()
    shapes = json.loads((root / "batch_shapes.json").read_text(encoding="utf-8"))
    assert shapes == [[2, 3, 16, 16], [2, 3, 20, 24]]
    payload = torch.load(root / "checkpoints" / "best.pt", map_location="cpu", weights_only=True)
    assert tuple(payload["chip_hw"]) == (16, 16)
    assert tuple(payload["tile_hw"]) == (20, 24)
    assert payload["gsd_m"] == pytest.approx(10.0)
    assert payload["chip_weight"] == pytest.approx(1.5)
    assert payload["tile_weight"] == pytest.approx(0.5)
    assert payload["in_channels"] == 3
    assert payload["arch"] == "dilatenet"
    assert payload["input_height_px"] == 20
    assert payload["input_width_px"] == 24
    assert payload["band_names"] == ["b0", "b1", "b2"]
    assert payload["dataset_hash"]
    assert payload["chip_dataset_hash"]
    assert "state_dict" in payload
    assert payload["epoch"] == 1
    assert "frame_hw" not in payload
    assert "window_px" not in payload
    last = torch.load(root / "checkpoints" / "last.pt", map_location="cpu", weights_only=True)
    assert tuple(last["chip_hw"]) == (16, 16)
    assert tuple(last["tile_hw"]) == (20, 24)
    assert "frame_hw" not in last
    assert "window_px" not in last
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert summary["dataset_hash"] == payload["dataset_hash"]
    assert summary["val_metric"] == "mean_dice"
    assert summary["loss"] == "focal_dice"
    assert summary["best_val_metric"] is not None
    assert summary["chip_hw"] == [16, 16]
    assert summary["tile_hw"] == [20, 24]
    assert summary["gsd_m"] == pytest.approx(10.0)
    assert summary["chip_weight"] == pytest.approx(1.5)
    assert summary["tile_weight"] == pytest.approx(0.5)
    assert summary["n_train_tile"] == 3
    assert summary["n_val_tile"] == 1
    assert summary["n_train"] == 3
    assert "frame_hw" not in summary
    assert "window_px" not in summary
    assert "test_mean_dice" not in summary
    with (root / "history.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    sources = {(row["split"], row["source"]) for row in rows}
    assert ("train", "chip") in sources
    assert ("train", "tile") in sources
    assert ("val", "chip") in sources
    assert ("val", "tile") in sources
    loaded = load_train_config(str(root / "config.toml"))
    assert loaded.chip_dir == str(chip)
    assert loaded.tile_dir == str(tile)
    assert loaded.chip_weight == pytest.approx(1.5)
    assert loaded.tile_weight == pytest.approx(0.5)
    assert loaded.max_steps == 2
    toml_text = (root / "config.toml").read_text(encoding="utf-8")
    assert "frame_hw" not in toml_text
    assert "window_px" not in toml_text
    assert "[canvas]" not in toml_text


def test_mixed_loop_uses_union_split(tmp_path: Path) -> None:
    """Row counts follow union_location_split, not the pack split files."""
    chip = tmp_path / "chips"
    tile = tmp_path / "tiles"
    _write_pack(chip, (16, 16))
    _write_pack(tile, (20, 24))
    chip_pack = load_processed_pack(chip)
    tile_pack = load_processed_pack(tile)
    assert chip_pack.splits.train == (0, 1, 4)
    located = union_location_split(chip_pack, tile_pack, recipe=SplitRecipe(seed=0))
    assert located.right.train == (0, 2, 4)
    assert located.left.train == located.right.train
    root = train(_config(chip, tile, tmp_path, kind="segmentor", run_id="union"))
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_train_chip"] == len(located.left.train)
    assert summary["n_train_tile"] == len(located.right.train)
    assert summary["n_val_chip"] == len(located.left.val)
    assert summary["n_val_tile"] == len(located.right.val)
    assert summary["n_test_chip"] == len(located.left.test)
    assert summary["n_test_tile"] == len(located.right.test)


def test_mixed_classifier_selects_tile_f1(tmp_path: Path) -> None:
    """A classifier mixed run selects the checkpoint with tile F1."""
    chip = tmp_path / "chips"
    tile = tmp_path / "tiles"
    _write_pack(chip, (16, 16))
    _write_pack(tile, (20, 24))
    root = train(_config(chip, tile, tmp_path, kind="classifier", run_id="mixed-cls"))
    shapes = json.loads((root / "batch_shapes.json").read_text(encoding="utf-8"))
    assert shapes == [[2, 3, 16, 16], [2, 3, 20, 24]]
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert summary["val_metric"] == "f1"
    assert summary["loss"] == "bce+focal_dice"
    assert summary["arch"] == "pactnet"
    payload = torch.load(root / "checkpoints" / "best.pt", map_location="cpu", weights_only=True)
    assert tuple(payload["chip_hw"]) == (16, 16)
    assert tuple(payload["tile_hw"]) == (20, 24)
    assert "frame_hw" not in payload
    assert "window_px" not in payload


def test_mixed_channel_mismatch_raises(tmp_path: Path) -> None:
    """A pack channel count that differs from in_channels raises."""
    chip = tmp_path / "chips"
    tile = tmp_path / "tiles"
    _write_pack(chip, (16, 16))
    _write_pack(tile, (20, 24))
    config = _config(chip, tile, tmp_path, kind="segmentor", run_id="bad", channels=4)
    with pytest.raises(ValueError, match="in_channels=4"):
        train(config)


def test_mixed_requires_both_directories(tmp_path: Path) -> None:
    """One pack directory is not a mixed-extent run."""
    with pytest.raises(ValueError, match="chip_dir and tile_dir"):
        train(
            TrainConfig(
                chip_dir=str(tmp_path / "chips"),
                run_dir=str(tmp_path / "runs"),
                run_id="half",
                device="cpu",
            )
        )


def test_source_weight_must_be_positive() -> None:
    """A non-positive source weight fails before the optimizer starts."""
    with pytest.raises(ValueError, match="chip_weight"):
        train(TrainConfig(chip_weight=0.0, device="cpu", epochs=1))
