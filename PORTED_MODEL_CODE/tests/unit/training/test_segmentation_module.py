"""Tests for SegmentationModule."""

from __future__ import annotations

import pytest
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from smoke_detection.training.segmentation_module import SegmentationModule


@pytest.fixture
def module():
    return SegmentationModule(in_channels=4, n_classes=1, bilinear=True, lr=1e-3)


def test_init_saves_hparams(module):
    assert module.hparams.lr == 1e-3


def test_forward_shape_matches_mask(module, tiny_segmentation_batch):
    out = module(tiny_segmentation_batch["img"])
    assert out.shape == tiny_segmentation_batch["fpt"].unsqueeze(1).shape


def test_shared_step_reshapes_mask_to_4d(module, tiny_segmentation_batch):
    loss, logits, y = module._shared_step(tiny_segmentation_batch)
    assert y.shape == (2, 1, 90, 90)
    assert logits.shape == (2, 1, 90, 90)
    assert loss.ndim == 0
    assert loss.requires_grad


def test_img_level_presence_logic(module):
    """Any-pixel true-positive counts as image-level hit."""
    import torch

    preds = torch.tensor([[[[0, 0], [0, 1]]], [[[0, 0], [0, 0]]]], dtype=torch.int32)
    y = torch.tensor([[[[0, 0], [0, 1]]], [[[0, 0], [0, 0]]]], dtype=torch.int32)
    image_pred = (preds.sum(dim=(1, 2, 3)) > 0).int()
    image_true = (y.sum(dim=(1, 2, 3)) > 0).int()
    assert image_pred.tolist() == [1, 0]
    assert image_true.tolist() == [1, 0]


def test_configure_optimizers_none(module):
    opt = module.configure_optimizers()
    assert isinstance(opt, torch.optim.Optimizer)


def test_configure_optimizers_plateau():
    m = SegmentationModule(lr=1e-3, scheduler="plateau")
    d = m.configure_optimizers()
    assert isinstance(d["lr_scheduler"]["scheduler"], ReduceLROnPlateau)


def test_configure_optimizers_cosine():
    m = SegmentationModule(lr=1e-3, scheduler="cosine")
    d = m.configure_optimizers()
    assert isinstance(d["lr_scheduler"], CosineAnnealingLR)


@pytest.mark.slow
def test_overfitting_sanity_drops_loss(tiny_segmentation_batch):
    m = SegmentationModule(in_channels=4, n_classes=1, bilinear=True, lr=1e-2)
    m.train()
    opt = torch.optim.SGD(m.parameters(), lr=1e-2, momentum=0.9)

    loss0, _, _ = m._shared_step(tiny_segmentation_batch)
    initial = float(loss0.item())
    for _ in range(20):
        opt.zero_grad()
        loss, _, _ = m._shared_step(tiny_segmentation_batch)
        loss.backward()
        opt.step()
    final = float(loss.item())
    assert final < initial
    assert final < 0.95 * initial
