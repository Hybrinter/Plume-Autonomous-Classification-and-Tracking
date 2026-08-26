"""Tests for ClassificationModule."""

from __future__ import annotations

import pytest
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from smoke_detection.training.classification_module import ClassificationModule


@pytest.fixture
def module():
    return ClassificationModule(in_channels=4, pretrained=False, lr=1e-3)


def test_init_saves_hparams(module):
    assert module.hparams.lr == 1e-3
    assert module.hparams.in_channels == 4


def test_forward_shape(module, tiny_classification_batch):
    out = module(tiny_classification_batch["img"])
    assert out.shape == (2, 1)


def test_shared_step_returns_scalar_loss(module, tiny_classification_batch):
    loss, logits, y = module._shared_step(tiny_classification_batch)
    assert loss.ndim == 0
    assert loss.requires_grad
    assert logits.shape == (2, 1)
    assert y.shape == (2, 1)


def test_configure_optimizers_none(module):
    opt = module.configure_optimizers()
    assert isinstance(opt, torch.optim.Optimizer)


def test_configure_optimizers_plateau():
    m = ClassificationModule(pretrained=False, lr=1e-3, scheduler="plateau")
    d = m.configure_optimizers()
    assert isinstance(d["lr_scheduler"]["scheduler"], ReduceLROnPlateau)
    assert d["lr_scheduler"]["monitor"] == "val/loss"


def test_configure_optimizers_cosine():
    m = ClassificationModule(pretrained=False, lr=1e-3, scheduler="cosine")
    d = m.configure_optimizers()
    assert isinstance(d["lr_scheduler"], CosineAnnealingLR)


@pytest.mark.slow
def test_overfitting_sanity_drops_loss(tiny_classification_batch):
    m = ClassificationModule(in_channels=4, pretrained=False, lr=1e-2)
    m.train()
    opt = torch.optim.SGD(m.parameters(), lr=1e-2, momentum=0.9)

    loss0, _, _ = m._shared_step(tiny_classification_batch)
    initial = float(loss0.item())
    for _ in range(20):
        opt.zero_grad()
        loss, _, _ = m._shared_step(tiny_classification_batch)
        loss.backward()
        opt.step()
    final = float(loss.item())
    assert final < initial
    assert final < 0.95 * initial
