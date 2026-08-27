"""Architecture smoke tests (skip without torch)."""

import importlib.util

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch extra not installed",
)


def test_unet_preserves_spatial_size() -> None:
    """U-Net maps (N, 4, H, W) to (N, 1, H, W) logits for more than one size."""
    import torch
    from tools.inference.arch.unet import UNet, build_segmentor

    net = build_segmentor().eval()
    assert isinstance(net, UNet)
    for size in (32, 64, 256):
        x = torch.zeros(1, 4, size, size)
        with torch.no_grad():
            y = net(x)
        assert y.shape == (1, 1, size, size)


def test_classifier_emits_one_logit() -> None:
    """ResNet-50 4-channel stem maps (N, 4, H, W) to (N, 1)."""
    import torch
    from tools.inference.arch.classifier import build_classifier

    net = build_classifier().eval()
    x = torch.zeros(2, 4, 256, 256)
    with torch.no_grad():
        y = net(x)
    assert y.shape == (2, 1)
    conv1 = net.conv1
    assert isinstance(conv1, torch.nn.Conv2d)
    assert conv1.in_channels == 4
    fc = net.fc
    assert isinstance(fc, torch.nn.Linear)
    assert fc.out_features == 1
