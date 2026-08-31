"""Shared 3x3 convolution primitives for the architecture catalog.

Dense blocks are one 3x3 convolution. Separable blocks are a depthwise 3x3
followed by a pointwise 1x1. Compact and dilated families insert batch-norm
and ReLU between those two convolutions. The scratch U-Net does not; it applies
batch-norm and ReLU after the pair.

Contains:
  - conv3x3_layers: convolution layers of one 3x3 stage, optional mid-norm.
  - conv_norm_relu: those layers plus a trailing batch-norm and ReLU.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

from torch import nn


def conv3x3_layers(
    in_channels: int,
    out_channels: int,
    *,
    stride: int = 1,
    dilation: int = 1,
    separable: bool = False,
    mid_norm: bool = False,
) -> list[nn.Module]:
    """Return the convolution layers of one 3x3 stage.

    Args:
        in_channels: Input feature depth.
        out_channels: Output feature depth.
        stride: Convolution stride.
        dilation: Kernel dilation. Padding equals dilation so spatial size is
            unchanged at stride 1.
        separable: Depthwise 3x3 plus pointwise 1x1.
        mid_norm: When ``separable`` is set, insert batch-norm and ReLU between
            the depthwise and pointwise convolutions.

    Returns:
        list[nn.Module]: Convolution layers, without a trailing norm on
        ``out_channels``. Bias is off; a later batch-norm supplies the shift.

    Notes:
        ``mid_norm`` has no effect on a dense 3x3. Compact and dilated blocks
        pass ``mid_norm=True`` with separable convolutions. The U-Net
        ``ConvBlock`` passes ``mid_norm=False``.
    """
    if not separable:
        return [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=dilation,
                dilation=dilation,
                bias=False,
            )
        ]
    layers: list[nn.Module] = [
        nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
    ]
    if mid_norm:
        layers += [nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True)]
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))
    return layers


def conv_norm_relu(
    in_channels: int,
    out_channels: int,
    *,
    stride: int = 1,
    dilation: int = 1,
    separable: bool = False,
    mid_norm: bool = False,
) -> nn.Sequential:
    """Return one 3x3 stage with a trailing batch-norm and ReLU.

    Args:
        in_channels: Input feature depth.
        out_channels: Output feature depth.
        stride: Convolution stride.
        dilation: Kernel dilation.
        separable: Depthwise 3x3 plus pointwise 1x1.
        mid_norm: Forwarded to :func:`conv3x3_layers`.

    Returns:
        nn.Sequential: Convolution layers, then ``BatchNorm2d(out_channels)``
        and ``ReLU``.
    """
    layers = conv3x3_layers(
        in_channels,
        out_channels,
        stride=stride,
        dilation=dilation,
        separable=separable,
        mid_norm=mid_norm,
    )
    layers += [nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)
