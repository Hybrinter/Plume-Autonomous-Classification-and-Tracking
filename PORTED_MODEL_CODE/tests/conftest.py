"""Top-level pytest fixtures.

Provides determinism, path helpers, a session-scoped synthetic prepared-dataset
tree (13-band uint16 GeoTIFFs + Label-Studio-style JSON polygons), and a
cached-weights fixture for the one slow test that loads ResNet-50 ImageNet
weights. Also auto-skips `gpu` tests when CUDA is unavailable.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests._data import build_synthetic_prepared_tree

REPO_ROOT = Path(__file__).resolve().parents[1]


def pytest_configure(config: pytest.Config) -> None:
    os.environ.setdefault("PYTORCH_LIGHTNING_LOG_LEVEL", "ERROR")
    # Lightning writes TB logs to CWD by default; tests redirect via cfg.paths.output_dir.


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-skip `gpu`-marked tests when CUDA is unavailable."""
    import torch

    if torch.cuda.is_available():
        return
    skip_gpu = pytest.mark.skip(reason="CUDA not available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


@pytest.fixture(autouse=True)
def _deterministic() -> None:
    """Seed all RNGs before every test."""
    from smoke_detection.common.seed import seed_everything

    seed_everything(1234, deterministic=True)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def synthetic_dataset_root(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Read-only session-scoped synthetic prepared-dataset tree."""
    root = tmp_path_factory.mktemp("smokedet_synthetic_prepared")
    build_synthetic_prepared_tree(root)
    prev = os.environ.get("SMOKEDET_DATA_ROOT")
    os.environ["SMOKEDET_DATA_ROOT"] = str(root)
    try:
        yield root
    finally:
        if prev is None:
            os.environ.pop("SMOKEDET_DATA_ROOT", None)
        else:
            os.environ["SMOKEDET_DATA_ROOT"] = prev
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
def mutable_dataset_root(tmp_path: Path) -> Path:
    """Function-scoped writable copy of the synthetic prepared tree."""
    root = tmp_path / "mutable_prepared"
    build_synthetic_prepared_tree(root)
    return root


@pytest.fixture
def tiny_classification_batch():
    """2-sample classification batch (pre-transform-compatible)."""
    import torch

    return {
        "idx": torch.tensor([0, 1]),
        "img": torch.randn(2, 4, 90, 90),
        "lbl": torch.tensor([True, False]),
        "imgfile": ["a.tif", "b.tif"],
    }


@pytest.fixture
def tiny_segmentation_batch():
    """2-sample segmentation batch (pre-transform-compatible)."""
    import torch

    mask = torch.zeros(2, 90, 90)
    mask[0, 20:50, 30:70] = 1.0  # one positive, one all-zero
    return {
        "idx": torch.tensor([0, 1]),
        "img": torch.randn(2, 4, 90, 90),
        "fpt": mask,
        "imgfile": ["a.tif", "b.tif"],
    }


@pytest.fixture(scope="session")
def cached_resnet50_weights():
    """Return IMAGENET1K_V1 weights enum; trigger download on first call."""
    from torchvision.models import ResNet50_Weights, resnet50

    _ = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    return ResNet50_Weights.IMAGENET1K_V1


@pytest.fixture
def sample_classification_config(synthetic_dataset_root: Path, tmp_path: Path):
    from smoke_detection.configs.classification import ClassificationConfig

    return ClassificationConfig.model_validate(
        {
            "task": "classification",
            "seed": 1234,
            "trainer": {
                "max_epochs": 1,
                "accelerator": "cpu",
                "devices": 1,
                "precision": "32",
                "deterministic": True,
                "log_every_n_steps": 1,
                "fast_dev_run": True,
            },
            "paths": {
                "data_root": str(synthetic_dataset_root),
                "output_dir": str(tmp_path / "lightning_logs"),
                "experiment_name": "test_classification",
            },
            "optim": {"lr": 1e-3, "momentum": 0.9, "weight_decay": 0.0, "scheduler": "none"},
            "model": {"backbone": "resnet50", "pretrained": False, "in_channels": 4},
            "data": {"batch_size": 2, "num_workers": 0, "crop_size": 90, "balance": "none"},
        }
    )


@pytest.fixture
def sample_segmentation_config(synthetic_dataset_root: Path, tmp_path: Path):
    from smoke_detection.configs.segmentation import SegmentationConfig

    return SegmentationConfig.model_validate(
        {
            "task": "segmentation",
            "seed": 1234,
            "trainer": {
                "max_epochs": 1,
                "accelerator": "cpu",
                "devices": 1,
                "precision": "32",
                "deterministic": True,
                "log_every_n_steps": 1,
                "fast_dev_run": True,
            },
            "paths": {
                "data_root": str(synthetic_dataset_root),
                "output_dir": str(tmp_path / "lightning_logs"),
                "experiment_name": "test_segmentation",
            },
            "optim": {"lr": 1e-3, "momentum": 0.9, "weight_decay": 0.0, "scheduler": "none"},
            "model": {"architecture": "unet", "in_channels": 4, "n_classes": 1, "bilinear": True},
            "data": {"batch_size": 2, "num_workers": 0, "crop_size": 90},
        }
    )


@pytest.fixture
def classification_yaml_tmp(sample_classification_config, tmp_path):
    import yaml

    p = tmp_path / "cls.yaml"
    p.write_text(yaml.safe_dump(sample_classification_config.model_dump(mode="json")))
    return p


@pytest.fixture
def segmentation_yaml_tmp(sample_segmentation_config, tmp_path):
    import yaml

    p = tmp_path / "seg.yaml"
    p.write_text(yaml.safe_dump(sample_segmentation_config.model_dump(mode="json")))
    return p
