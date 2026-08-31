"""Shared convolution primitive tests."""

from tools.inference.arch.blocks import conv3x3_layers, conv_norm_relu
from torch import nn


def test_dense_conv3x3_is_one_layer() -> None:
    """A dense stage is a single 3x3 convolution."""
    layers = conv3x3_layers(4, 8)
    assert len(layers) == 1
    conv = layers[0]
    assert isinstance(conv, nn.Conv2d)
    assert conv.kernel_size == (3, 3)
    assert conv.in_channels == 4
    assert conv.out_channels == 8


def test_separable_mid_norm_inserts_bn_relu() -> None:
    """Compact and dilated separable stages put BN-ReLU between dw and pw."""
    layers = conv3x3_layers(8, 16, separable=True, mid_norm=True)
    assert [type(layer) for layer in layers] == [nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.Conv2d]
    depthwise = layers[0]
    pointwise = layers[-1]
    assert isinstance(depthwise, nn.Conv2d)
    assert isinstance(pointwise, nn.Conv2d)
    assert depthwise.groups == 8
    assert pointwise.kernel_size == (1, 1)


def test_separable_without_mid_norm_is_dw_then_pw() -> None:
    """U-Net separable stages apply the pointwise convolution immediately."""
    layers = conv3x3_layers(8, 16, separable=True, mid_norm=False)
    assert [type(layer) for layer in layers] == [nn.Conv2d, nn.Conv2d]
    depthwise = layers[0]
    pointwise = layers[1]
    assert isinstance(depthwise, nn.Conv2d)
    assert isinstance(pointwise, nn.Conv2d)
    assert depthwise.groups == 8
    assert pointwise.kernel_size == (1, 1)


def test_conv_norm_relu_appends_bn_relu() -> None:
    """The trailing batch-norm and ReLU sit on the output channels."""
    block = conv_norm_relu(4, 8, separable=True, mid_norm=True)
    types = [type(module) for module in block]
    assert types[-2:] == [nn.BatchNorm2d, nn.ReLU]
    last_norm = block[-2]
    assert isinstance(last_norm, nn.BatchNorm2d)
    assert last_norm.num_features == 8
