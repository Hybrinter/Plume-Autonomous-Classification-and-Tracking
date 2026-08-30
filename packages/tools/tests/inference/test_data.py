"""Synthetic and disk-adapter batch tests."""

from pathlib import Path

import numpy as np
import pytest
import torch
from tools.inference.data import (
    SplitDataset,
    apply_train_augment,
    load_disk_batch,
    load_processed_pack,
    load_split,
    make_synthetic_batch,
    make_synthetic_pack,
    write_processed_pack,
)
from tools.inference.split import SplitRecipe


def test_synthetic_segmentor_plants_a_blob() -> None:
    """Segmentor synthetic masks have a positive rectangle and matching image lift."""
    batch = make_synthetic_batch("segmentor", 2, 4, 16, 16, seed=0)
    assert isinstance(batch.images, torch.Tensor)
    assert batch.images.shape == (2, 4, 16, 16)
    assert batch.targets.shape == (2, 1, 16, 16)
    assert batch.images.dtype == torch.float32
    assert batch.images.device.type == "cpu"
    assert float(batch.targets.max()) == 1.0
    assert float(batch.images[:, :, 8, 8].mean()) > float(batch.images[:, :, 0, 0].mean())


def test_synthetic_classifier_labels_even_indices() -> None:
    """Classifier synthetic even samples are positive."""
    batch = make_synthetic_batch("classifier", 4, 4, 8, 8, seed=1)
    assert batch.targets.shape == (4, 1)
    assert float(batch.targets[0, 0]) == 1.0
    assert float(batch.targets[1, 0]) == 0.0


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
    assert float(batch.targets[0, 0]) == 1.0


def test_disk_adapter_normalizes_dn(tmp_path: Path) -> None:
    """DN-valued images pass through normalize_dn."""
    images = np.full((1, 4, 4, 4), 4095.0, dtype=np.float32)
    masks = np.ones((1, 4, 4), dtype=np.float32)
    np.save(tmp_path / "images.npy", images)
    np.save(tmp_path / "masks.npy", masks)
    batch = load_disk_batch(tmp_path, "segmentor", bit_depth=12)
    assert float(batch.images.max()) == pytest.approx(1.0)
    assert batch.targets.shape == (1, 1, 4, 4)


def test_synthetic_pack_labels_follow_masks() -> None:
    """Even synthetic pack indices are positive for both heads."""
    images, masks, labels = make_synthetic_pack(4, 4, 8, 8, seed=0)
    assert images.shape == (4, 4, 8, 8)
    assert masks.shape == (4, 1, 8, 8)
    assert labels.shape == (4, 1)
    assert float(labels[0, 0]) == 1.0
    assert float(labels[1, 0]) == 0.0
    assert float(masks[0].max()) == 1.0
    assert float(masks[1].max()) == 0.0


def test_write_and_load_split(tmp_path: Path) -> None:
    """load_split returns Dataset views of the train/val/test subsets."""
    images, masks, labels = make_synthetic_pack(6, 4, 8, 8, seed=0)
    meta = write_processed_pack(
        tmp_path,
        images=images,
        masks=masks,
        labels=labels,
        recipe=SplitRecipe(seed=0),
        source_doi="synthetic",
    )
    pack = load_processed_pack(tmp_path)
    assert pack.meta.dataset_hash == meta.dataset_hash
    assert isinstance(pack.images, torch.Tensor)
    train = load_split(tmp_path, "segmentor", "train")
    val = load_split(tmp_path, "classifier", "val")
    test = load_split(tmp_path, "segmentor", "test")
    assert isinstance(train, SplitDataset)
    image, mask = train[0]
    assert image.shape == (4, 8, 8)
    assert mask.shape == (1, 8, 8)
    _, label = val[0]
    assert label.shape == (1,)
    total = len(train) + len(val) + len(test)
    assert total == 6


def test_load_processed_pack_skips_masks(tmp_path: Path) -> None:
    """load_masks=False stores a dummy mask tensor and still yields labels."""
    images, masks, labels = make_synthetic_pack(4, 4, 8, 8, seed=3)
    write_processed_pack(tmp_path, images, masks, labels, SplitRecipe(seed=3))
    pack = load_processed_pack(tmp_path, load_masks=False)
    assert pack.labels.shape == (4, 1)
    assert pack.masks.shape == (4, 1, 1, 1)
    full = load_processed_pack(tmp_path, load_masks=True)
    assert full.masks.shape == (4, 1, 8, 8)
    """A tampered labels.npy fails the dataset.json hash check."""
    images, masks, labels = make_synthetic_pack(4, 4, 8, 8, seed=1)
    write_processed_pack(tmp_path, images, masks, labels, SplitRecipe(seed=1))
    tampered = labels.detach().cpu().numpy().copy()
    tampered[0, 0] = 1.0 - float(tampered[0, 0])
    np.save(tmp_path / "labels.npy", tampered)
    with pytest.raises(ValueError, match="hash"):
        load_processed_pack(tmp_path)


def test_apply_train_augment_is_deterministic() -> None:
    """Same seed and index yield the same flip and rotation."""
    image = torch.arange(4 * 8 * 8, dtype=torch.float32).reshape(4, 8, 8)
    mask = torch.zeros(1, 8, 8, dtype=torch.float32)
    mask[0, 0:2, 0:2] = 1.0
    first_img, first_mask = apply_train_augment(image, mask, "segmentor", seed=0, index=3)
    second_img, second_mask = apply_train_augment(image, mask, "segmentor", seed=0, index=3)
    assert torch.equal(first_img, second_img)
    assert torch.equal(first_mask, second_mask)
    label = torch.tensor([1.0], dtype=torch.float32)
    _, out_label = apply_train_augment(image, label, "classifier", seed=0, index=3)
    assert torch.equal(out_label, label)


def test_load_split_rejects_unknown_kind(tmp_path: Path) -> None:
    """Unknown kind raises ValueError."""
    images, masks, labels = make_synthetic_pack(4, 4, 8, 8, seed=2)
    write_processed_pack(tmp_path, images, masks, labels, SplitRecipe(seed=2))
    with pytest.raises(ValueError, match="unknown train kind"):
        load_split(tmp_path, "nope", "train")
