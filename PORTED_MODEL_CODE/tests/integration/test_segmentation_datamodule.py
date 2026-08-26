"""Integration: SegmentationDataModule end-to-end on synthetic data."""

from __future__ import annotations

import torch

from smoke_detection.data.segmentation_datamodule import SegmentationDataModule


def test_setup_fit_builds_train_and_val(synthetic_dataset_root):
    dm = SegmentationDataModule(
        data_root=synthetic_dataset_root,
        batch_size=2,
        num_workers=0,
    )
    dm.setup(stage="fit")
    assert len(dm.train_ds) > 0
    assert len(dm.val_ds) > 0


def test_setup_test_standalone(synthetic_dataset_root):
    dm = SegmentationDataModule(
        data_root=synthetic_dataset_root,
        batch_size=2,
        num_workers=0,
    )
    dm.setup(stage="test")
    assert hasattr(dm, "test_ds") and len(dm.test_ds) > 0


def test_train_batch_shapes(synthetic_dataset_root):
    dm = SegmentationDataModule(
        data_root=synthetic_dataset_root,
        batch_size=2,
        num_workers=0,
        crop_size=90,
    )
    dm.setup(stage="fit")
    batch = next(iter(dm.train_dataloader()))
    assert batch["img"].shape == (2, 4, 90, 90)
    assert batch["fpt"].shape == (2, 90, 90)
    assert batch["img"].dtype == torch.float32


def test_num_workers_zero_on_windows(monkeypatch, synthetic_dataset_root):
    import platform

    monkeypatch.setattr(platform, "system", lambda: "Windows")
    dm = SegmentationDataModule(
        data_root=synthetic_dataset_root,
        batch_size=2,
        num_workers=4,
    )
    assert dm.num_workers == 0
