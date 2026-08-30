"""Clean-room encoder-decoder segmentor (U-Net family).

This module is an independent rewrite. It does not copy third-party U-Net
sources. Spatial size is taken from the input tensor; height and width are not
frozen in the graph. Output is raw logits. Flight applies sigmoid after the
ONNX session.

Three axes are parameterised so the family can be searched for a size/quality
trade-off rather than fixed at one point:

  - ``base_width`` scales every stage. Parameter count is roughly quadratic in
    it, so halving the width is close to a 4x shrink.
  - ``depth`` sets how many encoder stages precede the bottleneck, which
    controls receptive field and the coarsest stride.
  - ``separable`` swaps each 3x3 convolution for a depthwise 3x3 followed by a
    pointwise 1x1, cutting a block's cost by roughly the kernel area while
    keeping the same channel geometry.

The defaults (``base_width=64``, ``depth=4``, dense convolutions) reproduce the
original 64-128-256-512 bilinear U-Net exactly.

Contains:
  - ENCODER_CHANNELS: the default 64-128-256-512 stage widths.
  - stage_widths: per-stage channel counts for a width and depth.
  - ConvBlock: two 3x3 conv-BN-ReLU stages with unchanged spatial size.
  - EncoderStage: 2x2 max-pool then ConvBlock.
  - DecoderStage: bilinear upsample, skip concat, ConvBlock.
  - UNet: parameterised encoder-decoder with a 1x1 logit head.
  - build_segmentor: construct a 4-channel, 1-logit U-Net.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

ENCODER_CHANNELS: tuple[int, int, int, int] = (64, 128, 256, 512)


def stage_widths(base_width: int, depth: int) -> tuple[int, ...]:
    """Return the per-stage channel counts for a width and depth.

    Args:
        base_width: Channel count of the stem.
        depth: Number of stages including the stem.

    Returns:
        tuple[int, ...]: ``base_width`` doubled at each stage, so
        ``stage_widths(64, 4)`` is ``ENCODER_CHANNELS``.

    Raises:
        ValueError: If ``base_width`` or ``depth`` is below one.
    """
    if base_width < 1:
        raise ValueError(f"base_width must be >= 1; got {base_width}")
    if depth < 1:
        raise ValueError(f"depth must be >= 1; got {depth}")
    return tuple(base_width * (2**i) for i in range(depth))


def _conv3x3(in_channels: int, out_channels: int, separable: bool) -> tuple[nn.Module, ...]:
    """Return the convolution layers of one 3x3 stage.

    Bias is omitted throughout because batch-norm follows every convolution.
    """
    if not separable:
        return (nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),)
    return (
        nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=False,
        ),
        nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
    )


class ConvBlock(nn.Module):
    """Two padded 3x3 convolutions with batch-norm and ReLU.

    Padding keeps height and width unchanged. Bias is off because batch-norm
    follows each convolution. When ``separable`` is set, each 3x3 becomes a
    depthwise 3x3 plus a pointwise 1x1.
    """

    def __init__(self, in_channels: int, out_channels: int, separable: bool = False) -> None:
        """Build the two-convolution block.

        Args:
            in_channels: Input feature depth.
            out_channels: Output feature depth.
            separable: Use depthwise-separable convolutions.
        """
        super().__init__()
        layers: list[nn.Module] = []
        layers.extend(_conv3x3(in_channels, out_channels, separable))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.extend(_conv3x3(out_channels, out_channels, separable))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the two convolution stages.

        Args:
            x: np-equivalent tensor (N, in_channels, H, W).

        Returns:
            torch.Tensor: (N, out_channels, H, W).
        """
        return cast(torch.Tensor, self.block(x))


class EncoderStage(nn.Module):
    """Downsample by 2 then extract features."""

    def __init__(self, in_channels: int, out_channels: int, separable: bool = False) -> None:
        """Build max-pool plus ConvBlock.

        Args:
            in_channels: Input feature depth.
            out_channels: Output feature depth.
            separable: Use depthwise-separable convolutions.
        """
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels, separable=separable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool then convolve.

        Args:
            x: (N, in_channels, H, W).

        Returns:
            torch.Tensor: (N, out_channels, H/2, W/2).
        """
        return cast(torch.Tensor, self.conv(self.pool(x)))


class DecoderStage(nn.Module):
    """Bilinear upsample, concatenate the skip tensor, then ConvBlock."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        separable: bool = False,
    ) -> None:
        """Build upsample-concat-conv.

        Args:
            in_channels: Channel count of the decoder tensor before upsample.
            skip_channels: Channel count of the encoder skip tensor.
            out_channels: Channel count after ConvBlock.
            separable: Use depthwise-separable convolutions.
        """
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels, separable=separable)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Upsample `x` to the skip spatial size, concat, and convolve.

        Args:
            x: Decoder tensor (N, in_channels, h, w).
            skip: Encoder skip tensor (N, skip_channels, H, W).

        Returns:
            torch.Tensor: (N, out_channels, H, W).

        Notes:
            Interpolate uses the skip height and width so odd sizes still align.
        """
        x_up = nn.functional.interpolate(
            x, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )
        return cast(torch.Tensor, self.conv(torch.cat((x_up, skip), dim=1)))


class UNet(nn.Module):
    """Bilinear U-Net with a single logit channel.

    A stem ConvBlock at ``base_width`` channels, ``depth - 1`` encoder stages
    that double the width each time, a bottleneck that holds the widest stage,
    then one decoder stage per skip tensor and a 1x1 convolution. Skip tensors
    are concatenated on the channel axis. The forward pass does not apply
    sigmoid.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        base_width: int = 64,
        depth: int = 4,
        separable: bool = False,
    ) -> None:
        """Construct the encoder, bottleneck, decoder, and logit head.

        Args:
            in_channels: Input band count (flight default 4).
            out_channels: Logit maps (flight default 1).
            base_width: Stem channel count. Stages double from here.
            depth: Stage count including the stem.
            separable: Use depthwise-separable convolutions throughout.

        Raises:
            ValueError: If ``base_width`` or ``depth`` is below one.
        """
        super().__init__()
        widths = stage_widths(base_width, depth)
        self.stem = ConvBlock(in_channels, widths[0], separable=separable)
        self.encoders = nn.ModuleList(
            [EncoderStage(widths[i - 1], widths[i], separable=separable) for i in range(1, depth)]
        )
        self.bottleneck = EncoderStage(widths[-1], widths[-1], separable=separable)
        decoders: list[DecoderStage] = []
        current = widths[-1]
        for i in range(depth - 1, -1, -1):
            out_width = widths[max(i - 1, 0)]
            decoders.append(DecoderStage(current, widths[i], out_width, separable=separable))
            current = out_width
        self.decoders = nn.ModuleList(decoders)
        self.head = nn.Conv2d(widths[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a preprocessed batch to a logit mask of the same spatial size.

        Args:
            x: (N, in_channels, H, W) float32 in the normalize_dn domain.

        Returns:
            torch.Tensor: (N, out_channels, H, W) logits.
        """
        skips: list[torch.Tensor] = [self.stem(x)]
        for encoder in self.encoders:
            skips.append(encoder(skips[-1]))
        current = self.bottleneck(skips[-1])
        for decoder, skip in zip(self.decoders, reversed(skips), strict=True):
            current = decoder(current, skip)
        return cast(torch.Tensor, self.head(current))


def build_segmentor(
    in_channels: int = 4,
    out_channels: int = 1,
    base_width: int = 64,
    depth: int = 4,
    separable: bool = False,
) -> UNet:
    """Return a U-Net segmentor.

    Args:
        in_channels: Input band count.
        out_channels: Logit maps.
        base_width: Stem channel count.
        depth: Stage count including the stem.
        separable: Use depthwise-separable convolutions.

    Returns:
        UNet: Untrained segmentor in train mode. The defaults reproduce the
        64-128-256-512 baseline.
    """
    return UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        base_width=base_width,
        depth=depth,
        separable=separable,
    )
