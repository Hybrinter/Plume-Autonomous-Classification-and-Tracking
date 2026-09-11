"""Train-loop tests."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from tools.inference.data import make_synthetic_pack, write_processed_pack
from tools.inference.split import SplitRecipe
from tools.inference.train import (
    TrainConfig,
    config_digest,
    fit_batch_size,
    is_cuda_oom,
    load_train_config,
    next_batch_after_oom,
    overlay_train_config,
    train,
)


def test_load_train_config_defaults() -> None:
    """load_train_config() without a file returns frozen defaults."""
    cfg = load_train_config()
    assert cfg.kind == "segmentor"
    assert cfg.input_height_px == 256
    assert cfg.input_width_px == 256
    assert cfg.run_dir == "artifacts/runs"
    assert cfg.weight_decay == 0.0


def test_load_train_config_toml(tmp_path: Path) -> None:
    """TOML overlays known fields including arch and run_id."""
    path = tmp_path / "train.toml"
    path.write_text('kind = "classifier"\nepochs = 3\narch = "resnet50"\n', encoding="utf-8")
    cfg = load_train_config(str(path))
    assert cfg.kind == "classifier"
    assert cfg.epochs == 3
    assert cfg.arch == "resnet50"
    assert cfg.input_height_px == 256


def test_overlay_train_config_cli() -> None:
    """CLI overlays replace only the provided fields."""
    cfg = overlay_train_config(TrainConfig(), kind="classifier", epochs=2, run_id="exp")
    assert cfg.kind == "classifier"
    assert cfg.epochs == 2
    assert cfg.run_id == "exp"
    assert cfg.batch_size == 2


def test_train_writes_run_directory(tmp_path: Path) -> None:
    """One SGD epoch writes history, last, best, config, and summary."""
    run_dir = tmp_path / "runs"
    root = train(
        TrainConfig(
            kind="segmentor",
            epochs=1,
            batch_size=2,
            synthetic_samples=4,
            input_height_px=32,
            input_width_px=32,
            run_dir=str(run_dir),
            run_id="seg-test",
            seed=0,
        )
    )
    assert root.is_dir()
    assert (root / "config.toml").is_file()
    assert (root / "history.csv").is_file()
    assert (root / "summary.json").is_file()
    last = root / "checkpoints" / "last.pt"
    best = root / "checkpoints" / "best.pt"
    assert last.is_file()
    assert best.is_file()
    payload = torch.load(last, map_location="cpu", weights_only=True)
    assert payload["kind"] == "segmentor"
    assert payload["arch"] == "dilatenet"
    assert payload["input_height_px"] == 32
    with (root / "history.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    splits = {row["split"] for row in rows}
    assert "train" in splits
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert summary["run_id"] == "seg-test"
    assert summary["n_train"] >= 1
    assert int(summary["n_params"]) > 0
    assert int(summary["flops"]) > 0
    assert int(summary["batch_size"]) == 2


def test_is_cuda_oom_detects_runtime_out_of_memory() -> None:
    """CUDA OOM is a RuntimeError that names out of memory, not every RuntimeError."""
    assert is_cuda_oom(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
    assert not is_cuda_oom(RuntimeError("shape mismatch"))
    assert not is_cuda_oom(ValueError("CUDA out of memory"))


def test_next_batch_after_oom_halves_and_stops_at_one() -> None:
    """A failed batch size halves; size 1 cannot shrink further."""
    assert next_batch_after_oom(8) == 4
    assert next_batch_after_oom(3) == 1
    with pytest.raises(RuntimeError, match="batch_size=1"):
        next_batch_after_oom(1)


def test_fit_batch_size_walks_down_until_attempt_accepts() -> None:
    """fit_batch_size keeps the first size whose attempt does not raise OOM."""
    seen: list[int] = []

    def attempt(batch: int) -> None:
        seen.append(batch)
        if batch > 2:
            raise RuntimeError("CUDA out of memory")

    assert fit_batch_size(8, attempt) == 2
    assert seen == [8, 4, 2]


def test_fit_batch_size_raises_when_size_one_still_ooms() -> None:
    """A probe that never fits raises at batch size 1 rather than looping."""

    def attempt(_batch: int) -> None:
        raise RuntimeError("CUDA out of memory")

    with pytest.raises(RuntimeError, match="batch_size=1"):
        fit_batch_size(4, attempt)


def test_train_one_step_classifier(tmp_path: Path) -> None:
    """One SGD epoch on a 32 px classifier writes a classifier checkpoint."""
    root = train(
        TrainConfig(
            kind="classifier",
            epochs=1,
            batch_size=2,
            synthetic_samples=4,
            input_height_px=32,
            input_width_px=32,
            run_dir=str(tmp_path / "runs"),
            seed=0,
        )
    )
    payload = torch.load(root / "checkpoints" / "last.pt", map_location="cpu", weights_only=True)
    assert payload["kind"] == "classifier"
    assert payload["arch"] == "pactnet"


def test_train_processed_pack_splits(tmp_path: Path) -> None:
    """Train reads train/val indices from a processed pack."""
    pack_dir = tmp_path / "pack"
    images, masks, labels = make_synthetic_pack(6, 4, 32, 32, seed=0)
    write_processed_pack(pack_dir, images, masks, labels, SplitRecipe(seed=0))
    root = train(
        TrainConfig(
            kind="segmentor",
            input_height_px=32,
            input_width_px=32,
            epochs=1,
            batch_size=2,
            data_dir=str(pack_dir),
            run_dir=str(tmp_path / "runs"),
            run_id="from-pack",
        )
    )
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_val"] >= 1
    assert summary["dataset_hash"]


def test_train_disk_adapter(tmp_path: Path) -> None:
    """Train reads a packed numpy directory when data_dir has no splits.json."""
    images = np.zeros((2, 4, 32, 32), dtype=np.float32)
    images[0, :, 8:24, 8:24] = 0.9
    masks = np.zeros((2, 32, 32), dtype=np.float32)
    masks[0, 8:24, 8:24] = 1.0
    np.save(tmp_path / "images.npy", images)
    np.save(tmp_path / "masks.npy", masks)
    extra = tmp_path / "from_disk.pt"
    root = train(
        TrainConfig(
            kind="segmentor",
            input_height_px=32,
            input_width_px=32,
            epochs=1,
            batch_size=2,
            data_dir=str(tmp_path),
            checkpoint_path=str(extra),
            run_dir=str(tmp_path / "runs"),
            run_id="disk",
        )
    )
    assert extra.is_file()
    assert (root / "checkpoints" / "last.pt").is_file()


def test_config_digest_changes_with_learning_rate() -> None:
    """config_digest changes when an experiment field changes."""
    base = TrainConfig(kind="segmentor", arch="unet", seed=0)
    other = overlay_train_config(base, learning_rate=0.001)
    assert config_digest(base) != config_digest(other)
    same_dir = overlay_train_config(base, run_dir="/tmp/other")
    assert config_digest(base) == config_digest(same_dir)


def test_train_default_run_id_includes_digest(tmp_path: Path) -> None:
    """Empty run_id writes {kind}-{arch}-{seed}-{digest8}."""
    cfg = TrainConfig(
        kind="segmentor",
        epochs=1,
        batch_size=2,
        synthetic_samples=4,
        input_height_px=32,
        input_width_px=32,
        run_dir=str(tmp_path / "runs"),
        seed=0,
    )
    root = train(cfg)
    digest = config_digest(cfg)
    assert root.name == f"segmentor-dilatenet-0-{digest}"


def test_train_refuses_existing_run(tmp_path: Path) -> None:
    """A second train on the same run_id raises FileExistsError."""
    cfg = TrainConfig(
        kind="segmentor",
        epochs=1,
        batch_size=2,
        synthetic_samples=4,
        input_height_px=32,
        input_width_px=32,
        run_dir=str(tmp_path / "runs"),
        run_id="same",
        seed=0,
    )
    train(cfg)
    with pytest.raises(FileExistsError, match="run directory exists"):
        train(cfg)


def test_train_overwrite_replaces(tmp_path: Path) -> None:
    """overwrite=True replaces an existing run directory."""
    cfg = TrainConfig(
        kind="segmentor",
        epochs=1,
        batch_size=2,
        synthetic_samples=4,
        input_height_px=32,
        input_width_px=32,
        run_dir=str(tmp_path / "runs"),
        run_id="same",
        seed=0,
    )
    first = train(cfg)
    second = train(overlay_train_config(cfg, overwrite=True))
    assert second == first
    assert (second / "summary.json").is_file()


def test_overlay_learning_rate_and_overwrite() -> None:
    """CLI overlays replace learning_rate and overwrite when set."""
    cfg = overlay_train_config(TrainConfig(), learning_rate=0.001, overwrite=True)
    assert cfg.learning_rate == 0.001
    assert cfg.overwrite is True
    assert cfg.momentum == 0.9


def test_train_adamw_cosine(tmp_path: Path) -> None:
    """AdamW plus cosine writes optimizer fields into summary.json."""
    root = train(
        TrainConfig(
            kind="segmentor",
            epochs=1,
            batch_size=2,
            synthetic_samples=4,
            input_height_px=32,
            input_width_px=32,
            run_dir=str(tmp_path / "runs"),
            run_id="adam",
            optimizer="adamw",
            scheduler="cosine",
        )
    )
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert summary["optimizer"] == "adamw"
    assert summary["scheduler"] == "cosine"


def test_train_shuffle_pos_weight_augment(tmp_path: Path) -> None:
    """shuffle, pos_weight, and augment complete one epoch."""
    root = train(
        TrainConfig(
            kind="segmentor",
            epochs=1,
            batch_size=2,
            synthetic_samples=4,
            input_height_px=32,
            input_width_px=32,
            run_dir=str(tmp_path / "runs"),
            run_id="loop-opts",
            shuffle=True,
            pos_weight=2.0,
            augment=True,
        )
    )
    assert (root / "summary.json").is_file()


def test_train_unknown_optimizer() -> None:
    """Unknown optimizer fails schema validation."""
    with pytest.raises(ValueError, match="optimizer"):
        TrainConfig(optimizer="nope")  # type: ignore[arg-type]
