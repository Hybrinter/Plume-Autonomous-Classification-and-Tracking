"""Train-loop tests. Torch-backed cases skip when the train extra is absent."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from tools.model.train import TrainConfig, load_train_config, overlay_train_config, train

_HAS_TORCH = importlib.util.find_spec("torch") is not None


def test_load_train_config_defaults() -> None:
    """load_train_config() without a file returns frozen defaults."""
    cfg = load_train_config()
    assert cfg.kind == "segmentor"
    assert cfg.input_height_px == 256
    assert cfg.input_width_px == 256


def test_load_train_config_toml(tmp_path: Path) -> None:
    """TOML overlays known fields."""
    path = tmp_path / "train.toml"
    path.write_text('kind = "classifier"\nepochs = 3\n', encoding="utf-8")
    cfg = load_train_config(str(path))
    assert cfg.kind == "classifier"
    assert cfg.epochs == 3
    assert cfg.input_height_px == 256


def test_overlay_train_config_cli() -> None:
    """CLI overlays replace only the provided fields."""
    cfg = overlay_train_config(TrainConfig(), kind="classifier", epochs=2)
    assert cfg.kind == "classifier"
    assert cfg.epochs == 2
    assert cfg.batch_size == 2


@pytest.mark.skipif(not _HAS_TORCH, reason="torch extra not installed")
def test_train_one_step_segmentor_256(tmp_path: Path) -> None:
    """One SGD step on synthetic (N, 4, 256, 256) writes a segmentor checkpoint."""
    import torch

    out = tmp_path / "seg.pt"
    path = train(
        TrainConfig(
            kind="segmentor",
            epochs=1,
            batch_size=2,
            synthetic_samples=2,
            checkpoint_path=str(out),
            seed=0,
        )
    )
    assert path.is_file()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["kind"] == "segmentor"
    assert payload["input_height_px"] == 256


@pytest.mark.skipif(not _HAS_TORCH, reason="torch extra not installed")
def test_train_one_step_classifier_256(tmp_path: Path) -> None:
    """One SGD step on synthetic (N, 4, 256, 256) writes a classifier checkpoint."""
    import torch

    out = tmp_path / "clf.pt"
    path = train(
        TrainConfig(
            kind="classifier",
            epochs=1,
            batch_size=2,
            synthetic_samples=2,
            checkpoint_path=str(out),
            seed=0,
        )
    )
    assert path.is_file()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["kind"] == "classifier"


@pytest.mark.skipif(not _HAS_TORCH, reason="torch extra not installed")
def test_train_disk_adapter(tmp_path: Path) -> None:
    """Train reads a packed numpy directory when data_dir is set."""
    images = np.zeros((2, 4, 32, 32), dtype=np.float32)
    images[0, :, 8:24, 8:24] = 0.9
    masks = np.zeros((2, 32, 32), dtype=np.float32)
    masks[0, 8:24, 8:24] = 1.0
    np.save(tmp_path / "images.npy", images)
    np.save(tmp_path / "masks.npy", masks)
    out = tmp_path / "from_disk.pt"
    path = train(
        TrainConfig(
            kind="segmentor",
            input_height_px=32,
            input_width_px=32,
            epochs=1,
            batch_size=2,
            data_dir=str(tmp_path),
            checkpoint_path=str(out),
        )
    )
    assert path.is_file()
