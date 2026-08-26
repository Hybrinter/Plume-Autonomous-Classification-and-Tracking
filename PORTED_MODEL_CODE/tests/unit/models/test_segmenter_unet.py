"""Tests for build_segmenter and UNet forward."""

from __future__ import annotations

import pytest
import torch

from smoke_detection.models.segmenter_unet import UNet, build_segmenter


def test_build_segmenter_returns_unet():
    net = build_segmenter(in_channels=4, n_classes=1)
    assert isinstance(net, UNet)


@pytest.mark.parametrize("crop", [90, 120])
def test_unet_size_preserving_bilinear(crop):
    net = build_segmenter(in_channels=4, n_classes=1, bilinear=True).eval()
    with torch.no_grad():
        out = net(torch.randn(2, 4, crop, crop))
    assert out.shape == (2, 1, crop, crop)


def test_unet_convtranspose_path():
    net = build_segmenter(in_channels=4, n_classes=1, bilinear=False).eval()
    with torch.no_grad():
        out = net(torch.randn(2, 4, 120, 120))
    assert out.shape == (2, 1, 120, 120)
