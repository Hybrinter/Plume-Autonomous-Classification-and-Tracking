"""Fresh stems accept a non-RGB channel count."""

from __future__ import annotations

import torch
from tools.original_dataset_analysis.models import DilateNet, ShuffleNetClassifier


def test_shufflenet_accepts_twelve_channels() -> None:
    """A 12-band stem maps a small stack to one logit."""
    net = ShuffleNetClassifier(12).eval()
    logits = net(torch.zeros(2, 12, 32, 32))
    assert logits.shape == (2, 1)


def test_shufflenet_trains_a_singleton_30px_batch() -> None:
    """A batch of one 30-pixel tile is a legal training step."""
    net = ShuffleNetClassifier(3).train()
    logits = net(torch.zeros(1, 3, 30, 30))
    assert logits.shape == (1, 1)
    logits.sum().backward()


def test_dilatenet_keeps_spatial_size() -> None:
    """A 3-band stem maps a stack to a full-resolution logit plane."""
    net = DilateNet(3).eval()
    logits = net(torch.zeros(1, 3, 40, 40))
    assert logits.shape == (1, 1, 40, 40)
