"""Band-count stem surgery tests."""

import pytest
import torch
from tools.inference.arch.stem import (
    adapt_conv_in_channels,
    remap_stem_weight,
    retarget_final_linear,
    retarget_first_conv,
)
from torch import nn
from torchvision.models import mobilenet_v3_small, resnet18


def _rgb_kernel(out_channels: int = 4, kernel_size: int = 3) -> torch.Tensor:
    """Return a deterministic RGB stem kernel for permutation checks."""
    return torch.arange(out_channels * 3 * kernel_size * kernel_size, dtype=torch.float32).reshape(
        out_channels, 3, kernel_size, kernel_size
    )


@pytest.mark.parametrize("in_channels", [1, 3, 4, 6])
def test_remap_stem_weight_output_shape(in_channels: int) -> None:
    """remap_stem_weight returns (out, in_channels, kh, kw)."""
    weight = _rgb_kernel()
    remapped = remap_stem_weight(weight, in_channels)
    assert remapped.shape == (weight.shape[0], in_channels, weight.shape[2], weight.shape[3])


def test_remap_stem_weight_pact_band_permutation() -> None:
    """RGB bands follow BLUE<-blue, GREEN<-green, RED<-red with rescale."""
    weight = _rgb_kernel()
    remapped = remap_stem_weight(weight, 3)
    assert torch.allclose(remapped[:, 0], weight[:, 2])
    assert torch.allclose(remapped[:, 1], weight[:, 1])
    assert torch.allclose(remapped[:, 2], weight[:, 0])


def test_remap_stem_weight_nir_column_is_rgb_mean() -> None:
    """The NIR band takes the mean RGB column scaled by 3/in_channels."""
    weight = _rgb_kernel()
    in_channels = 4
    remapped = remap_stem_weight(weight, in_channels)
    rgb_mean = weight.mean(dim=1, keepdim=True)
    scale = 3.0 / float(in_channels)
    assert torch.allclose(remapped[:, 3], rgb_mean.squeeze(1) * scale)


def test_remap_stem_weight_rescale_by_in_channels() -> None:
    """Three bands permute exactly at scale 1.0; six bands halve the kernel."""
    weight = _rgb_kernel()
    three_band = remap_stem_weight(weight, 3)
    assert torch.allclose(three_band[:, 0], weight[:, 2])
    assert torch.allclose(three_band[:, 1], weight[:, 1])
    assert torch.allclose(three_band[:, 2], weight[:, 0])

    six_band = remap_stem_weight(weight, 6)
    assert torch.allclose(six_band[:, 0], weight[:, 2] * 0.5)
    assert torch.allclose(six_band[:, 1], weight[:, 1] * 0.5)
    assert torch.allclose(six_band[:, 2], weight[:, 0] * 0.5)


def test_remap_stem_weight_rejects_invalid_inputs() -> None:
    """Non-rank-4 kernels, non-RGB channel counts, and in_channels<1 raise."""
    rgb = _rgb_kernel()
    with pytest.raises(ValueError, match="expected an"):
        remap_stem_weight(rgb.reshape(4, 3, 9), 3)
    with pytest.raises(ValueError, match="expected an"):
        remap_stem_weight(torch.randn(4, 5, 3, 3), 4)
    with pytest.raises(ValueError, match="in_channels must be"):
        remap_stem_weight(rgb, 0)


def test_adapt_conv_in_channels_returns_same_when_matched() -> None:
    """No replacement is built when the band count already matches."""
    conv = nn.Conv2d(4, 8, kernel_size=3)
    assert adapt_conv_in_channels(conv, 4, pretrained=True) is conv


def test_adapt_conv_in_channels_preserves_geometry() -> None:
    """Replacement keeps kernel size, stride, padding, groups, and bias."""
    conv = nn.Conv2d(
        3,
        16,
        kernel_size=5,
        stride=2,
        padding=2,
        groups=1,
        bias=False,
    )
    adapted = adapt_conv_in_channels(conv, 4, pretrained=True)
    assert adapted.in_channels == 4
    assert adapted.out_channels == conv.out_channels
    assert adapted.kernel_size == conv.kernel_size
    assert adapted.stride == conv.stride
    assert adapted.padding == conv.padding
    assert adapted.groups == conv.groups
    assert adapted.bias is None


def test_adapt_conv_in_channels_pretrained_false_skips_weight_copy() -> None:
    """Random initialisation differs from a remapped pretrained kernel."""
    conv = nn.Conv2d(3, 8, kernel_size=3, bias=True)
    remapped = remap_stem_weight(conv.weight.detach(), 4)
    randomised = adapt_conv_in_channels(conv, 4, pretrained=False)
    assert not torch.allclose(randomised.weight, remapped)


@pytest.mark.parametrize("factory", [resnet18, mobilenet_v3_small])
def test_retarget_first_conv_accepts_four_bands(factory: type[nn.Module]) -> None:
    """Stem retargeting succeeds on ResNet and MobileNet attribute paths."""
    model = factory(weights=None).eval()
    retarget_first_conv(model, 4, pretrained=False)
    x = torch.zeros(1, 4, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out.ndim == 2


def test_retarget_final_linear_emits_one_logit() -> None:
    """Head retargeting maps backbone output to a single logit."""
    model = resnet18(weights=None).eval()
    retarget_final_linear(model, 1)
    x = torch.zeros(2, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1)


def test_retarget_helpers_reject_modules_without_target_layers() -> None:
    """Models without Conv2d or Linear stems raise ValueError."""
    relu_only = nn.Sequential(nn.ReLU())
    with pytest.raises(ValueError, match="no Conv2d"):
        retarget_first_conv(relu_only, 4, pretrained=False)
    with pytest.raises(ValueError, match="no Linear"):
        retarget_final_linear(relu_only, 1)
