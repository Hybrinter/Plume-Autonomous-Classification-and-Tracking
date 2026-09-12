"""Shared fixtures for tools.inference tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from tools.inference.train import TrainConfig, train


@pytest.fixture(scope="session")
def tiny_segmentor_ckpt(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One 32×32, 1-epoch segmentor checkpoint reused by export tests."""
    root = tmp_path_factory.mktemp("export_train_seg")
    ckpt = root / "seg.pt"
    train(
        TrainConfig(
            kind="segmentor",
            epochs=1,
            batch_size=2,
            synthetic_samples=4,
            input_height_px=32,
            input_width_px=32,
            checkpoint_path=str(ckpt),
            run_dir=str(root / "runs"),
            seed=0,
            device="cpu",
        )
    )
    return ckpt


@pytest.fixture(scope="session")
def tiny_classifier_ckpt(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One 32×32, 1-epoch classifier checkpoint reused by export tests."""
    root = tmp_path_factory.mktemp("export_train_cls")
    ckpt = root / "cls.pt"
    train(
        TrainConfig(
            kind="classifier",
            epochs=1,
            batch_size=2,
            synthetic_samples=4,
            input_height_px=32,
            input_width_px=32,
            checkpoint_path=str(ckpt),
            run_dir=str(root / "runs"),
            seed=0,
            device="cpu",
        )
    )
    return ckpt


@pytest.fixture(scope="session")
def tiny_dilatenet_ckpt(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One 32×32 dilatenet checkpoint reused by spatial export tests."""
    root = tmp_path_factory.mktemp("export_train_dilate")
    ckpt = root / "dilate.pt"
    train(
        TrainConfig(
            kind="segmentor",
            arch="dilatenet_w32",
            epochs=1,
            batch_size=2,
            synthetic_samples=4,
            input_height_px=32,
            input_width_px=32,
            checkpoint_path=str(ckpt),
            run_dir=str(root / "runs"),
            seed=0,
            device="cpu",
        )
    )
    return ckpt


@pytest.fixture(scope="session")
def tiny_dilatenet_16_ckpt(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One 16×16 dilatenet checkpoint for spatial re-export tests."""
    root = tmp_path_factory.mktemp("export_train_dilate16")
    ckpt = root / "dilate.pt"
    train(
        TrainConfig(
            kind="segmentor",
            arch="dilatenet_w32",
            epochs=1,
            batch_size=2,
            synthetic_samples=4,
            input_height_px=16,
            input_width_px=16,
            checkpoint_path=str(ckpt),
            run_dir=str(root / "runs"),
            seed=0,
            device="cpu",
        )
    )
    return ckpt
