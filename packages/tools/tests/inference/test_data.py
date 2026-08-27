"""Synthetic and disk-adapter batch tests (torch-free)."""

from pathlib import Path

import numpy as np
import pytest
from tools.inference.data import load_disk_batch, make_synthetic_batch


def test_synthetic_segmentor_plants_a_blob() -> None:
    """Segmentor synthetic masks have a positive rectangle and matching image lift."""
    batch = make_synthetic_batch("segmentor", 2, 4, 16, 16, seed=0)
    assert batch.images.shape == (2, 4, 16, 16)
    assert batch.targets.shape == (2, 1, 16, 16)
    assert float(batch.targets.max()) == 1.0
    assert float(batch.images[:, :, 8, 8].mean()) > float(batch.images[:, :, 0, 0].mean())


def test_synthetic_classifier_labels_even_indices() -> None:
    """Classifier synthetic even samples are positive."""
    batch = make_synthetic_batch("classifier", 4, 4, 8, 8, seed=1)
    assert batch.targets.shape == (4, 1)
    assert batch.targets[0, 0] == 1.0
    assert batch.targets[1, 0] == 0.0


def test_synthetic_rejects_unknown_kind() -> None:
    """Unknown kind raises ValueError."""
    with pytest.raises(ValueError):
        make_synthetic_batch("nope", 1, 4, 8, 8, seed=0)


def test_disk_adapter_classifier(tmp_path: Path) -> None:
    """load_disk_batch reads images.npy and labels.npy."""
    images = np.zeros((2, 4, 8, 8), dtype=np.float32)
    labels = np.array([1.0, 0.0], dtype=np.float32)
    np.save(tmp_path / "images.npy", images)
    np.save(tmp_path / "labels.npy", labels)
    batch = load_disk_batch(tmp_path, "classifier")
    assert batch.images.shape == (2, 4, 8, 8)
    assert batch.targets.shape == (2, 1)
    assert batch.targets[0, 0] == 1.0


def test_disk_adapter_normalizes_dn(tmp_path: Path) -> None:
    """DN-valued images pass through normalize_dn."""
    images = np.full((1, 4, 4, 4), 4095.0, dtype=np.float32)
    masks = np.ones((1, 4, 4), dtype=np.float32)
    np.save(tmp_path / "images.npy", images)
    np.save(tmp_path / "masks.npy", masks)
    batch = load_disk_batch(tmp_path, "segmentor", bit_depth=12)
    assert float(batch.images.max()) == pytest.approx(1.0)
    assert batch.targets.shape == (1, 1, 4, 4)
