"""Clean-room encoder-decoder segmentor (U-Net family).

This module is an independent rewrite. It does not copy third-party U-Net
sources. Channel widths follow the common 64-128-256-512 bilinear family.
Spatial size is taken from the input tensor; height and width are not frozen
in the graph.

Output is raw logits. Flight applies sigmoid after the ONNX session.

Contains:
  - ConvBlock: two 3x3 conv-BN-ReLU stages with unchanged spatial size.
  - EncoderStage: 2x2 max-pool then ConvBlock.
  - DecoderStage: bilinear upsample, skip concat, ConvBlock.
  - UNet: four-level encoder-decoder with a 1x1 logit head.
  - build_segmentor: construct a 4-channel, 1-logit U-Net.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn

ENCODER_CHANNELS: tuple[int, int, int, int] = (64, 128, 256, 512)

# Annotated Any so torch-free mypy (CI extra=dev, no torch) and the train extra
# both see a stable subclass target. nn.Module is a real class at runtime.
_ModuleBase: Any = nn.Module


class _TorchModule(_ModuleBase):  # type: ignore[misc]
    """Runtime nn.Module base for the U-Net stages."""


class ConvBlock(_TorchModule):
    """Two padded 3x3 convolutions with batch-norm and ReLU.

    Padding keeps height and width unchanged. Bias is off because batch-norm
    follows each convolution.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Build the two-convolution block.

        Args:
            in_channels: Input feature depth.
            out_channels: Output feature depth.
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the two convolution stages.

        Args:
            x: np-equivalent tensor (N, in_channels, H, W).

        Returns:
            torch.Tensor: (N, out_channels, H, W).
        """
        return cast(torch.Tensor, self.block(x))


class EncoderStage(_TorchModule):
    """Downsample by 2 then extract features."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Build max-pool plus ConvBlock.

        Args:
            in_channels: Input feature depth.
            out_channels: Output feature depth.
        """
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool then convolve.

        Args:
            x: (N, in_channels, H, W).

        Returns:
            torch.Tensor: (N, out_channels, H/2, W/2).
        """
        return cast(torch.Tensor, self.conv(self.pool(x)))


class DecoderStage(_TorchModule):
    """Bilinear upsample, concatenate the skip tensor, then ConvBlock."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        """Build upsample-concat-conv.

        Args:
            in_channels: Channel count of the decoder tensor before upsample.
            skip_channels: Channel count of the encoder skip tensor.
            out_channels: Channel count after ConvBlock.
        """
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

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


class UNet(_TorchModule):
    """Four-level bilinear U-Net with a single logit channel.

    Stem ConvBlock at 64 channels, three encoder stages to 512, a 512-channel
    bottleneck, four decoder stages back to 64, then a 1x1 convolution. Skip
    tensors are concatenated on the channel axis. The forward pass does not
    apply sigmoid.
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 1) -> None:
        """Construct the encoder, bottleneck, decoder, and logit head.

        Args:
            in_channels: Input band count (flight default 4).
            out_channels: Logit maps (flight default 1).
        """
        super().__init__()
        c1, c2, c3, c4 = ENCODER_CHANNELS
        self.stem = ConvBlock(in_channels, c1)
        self.enc1 = EncoderStage(c1, c2)
        self.enc2 = EncoderStage(c2, c3)
        self.enc3 = EncoderStage(c3, c4)
        self.bottleneck = EncoderStage(c4, c4)
        self.dec3 = DecoderStage(c4, c4, c3)
        self.dec2 = DecoderStage(c3, c3, c2)
        self.dec1 = DecoderStage(c2, c2, c1)
        self.dec0 = DecoderStage(c1, c1, c1)
        self.head = nn.Conv2d(c1, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a preprocessed batch to a logit mask of the same spatial size.

        Args:
            x: (N, in_channels, H, W) float32 in the normalize_dn domain.

        Returns:
            torch.Tensor: (N, out_channels, H, W) logits.
        """
        s0 = self.stem(x)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        b = self.bottleneck(s3)
        d3 = self.dec3(b, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        d0 = self.dec0(d1, s0)
        return cast(torch.Tensor, self.head(d0))


def build_segmentor(in_channels: int = 4, out_channels: int = 1) -> UNet:
    """Return a U-Net segmentor with the default channel family.

    Args:
        in_channels: Input band count.
        out_channels: Logit maps.

    Returns:
        UNet: Untrained segmentor in train mode.
    """
    return UNet(in_channels=in_channels, out_channels=out_channels)
