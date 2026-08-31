"""Architecture smoke tests."""

import torch
from tools.inference.arch.classifier import build_classifier
from tools.inference.arch.compact import PactNet
from tools.inference.arch.unet import UNet, build_segmentor


def test_unet_preserves_spatial_size() -> None:
    """U-Net maps (N, 4, H, W) to (N, 1, H, W) logits for more than one size."""
    net = build_segmentor().eval()
    assert isinstance(net, UNet)
    for size in (32, 64, 256):
        x = torch.zeros(1, 4, size, size)
        with torch.no_grad():
            y = net(x)
        assert y.shape == (1, 1, size, size)


def test_classifier_emits_one_logit() -> None:
    """Default pactnet maps (N, 4, H, W) to (N, 1)."""
    net = build_classifier().eval()
    assert isinstance(net, PactNet)
    x = torch.zeros(2, 4, 256, 256)
    with torch.no_grad():
        y = net(x)
    assert y.shape == (2, 1)
    assert isinstance(net.features, torch.nn.Sequential)
    assert isinstance(net.head, torch.nn.Linear)
    assert net.head.out_features == 1
    assert not hasattr(net, "conv1")
    assert not hasattr(net, "fc")
