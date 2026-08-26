"""Integration: LightningModule training_step on a real DataModule batch."""

from __future__ import annotations

import pytest
import torch

from smoke_detection.data.classification_datamodule import ClassificationDataModule
from smoke_detection.data.segmentation_datamodule import SegmentationDataModule
from smoke_detection.training.classification_module import ClassificationModule
from smoke_detection.training.segmentation_module import SegmentationModule

# self.log() is a no-op without an attached Trainer; these tests deliberately invoke
# training_step in isolation, so filter the expected "trainer not registered" notice.
pytestmark = pytest.mark.filterwarnings(
    "ignore:You are trying to `self.log\\(\\)`.*trainer.*not registered:UserWarning"
)


def test_classification_training_step_on_real_batch(synthetic_dataset_root):
    dm = ClassificationDataModule(
        data_root=synthetic_dataset_root,
        batch_size=2,
        num_workers=0,
        balance="none",
    )
    dm.setup(stage="fit")
    batch = next(iter(dm.train_dataloader()))
    module = ClassificationModule(in_channels=4, pretrained=False, lr=1e-3)
    loss = module.training_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.requires_grad


def test_segmentation_training_step_on_real_batch(synthetic_dataset_root):
    dm = SegmentationDataModule(
        data_root=synthetic_dataset_root,
        batch_size=2,
        num_workers=0,
    )
    dm.setup(stage="fit")
    batch = next(iter(dm.train_dataloader()))
    module = SegmentationModule(in_channels=4, n_classes=1, bilinear=True, lr=1e-3)
    loss = module.training_step(batch, 0)
    assert torch.isfinite(loss)
    assert loss.requires_grad
