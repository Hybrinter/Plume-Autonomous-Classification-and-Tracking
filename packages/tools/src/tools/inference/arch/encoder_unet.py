"""Segmentor with a torchvision ResNet encoder and a light U-Net decoder.

The scratch U-Net in :mod:`tools.inference.arch.unet` spends most of its
parameters on the encoder. Swapping in a ResNet encoder lets the same decoder
sit on top of ImageNet features, and lets encoder capacity and decoder capacity
be traded independently: ``decoder_width`` scales the decoder alone, so a
strong encoder can be paired with a decoder small enough to keep the exported
artifact modest.

Skip taps follow the ResNet stride schedule. For a 256 px input the encoder
emits features at 128, 64, 32, 16, and 8 px; the decoder walks back up and a
final bilinear resize restores the input size, so height and width stay out of
the graph exactly as they do for the scratch U-Net. Output is raw logits.

Contains:
  - RESNET_ENCODERS: registered encoder names.
  - encoder_tap_channels: skip channel counts for one encoder.
  - decoder_widths: decoder channel counts for a width knob.
  - ResNetUNet: encoder-decoder segmentor.
  - build_encoder_segmentor: construct one from a registry name.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    resnet18,
    resnet34,
    resnet50,
)

from tools.inference.arch.stem import retarget_first_conv
from tools.inference.arch.unet import ConvBlock

RESNET_ENCODERS: frozenset[str] = frozenset({"resnet18", "resnet34", "resnet50"})

_BASIC_TAPS: tuple[int, int, int, int, int] = (64, 64, 128, 256, 512)
_BOTTLENECK_TAPS: tuple[int, int, int, int, int] = (64, 256, 512, 1024, 2048)

_DECODER_STAGES = 5


def encoder_tap_channels(encoder: str) -> tuple[int, int, int, int, int]:
    """Return the skip channel counts a ResNet encoder emits.

    Args:
        encoder: Name in ``RESNET_ENCODERS``.

    Returns:
        tuple: Channels at strides 2, 4, 8, 16, and 32. ResNet-18 and
        ResNet-34 use BasicBlock widths; ResNet-50 uses Bottleneck widths,
        which are four times wider from ``layer1`` on.

    Raises:
        ValueError: If the encoder is not registered.
    """
    if encoder in {"resnet18", "resnet34"}:
        return _BASIC_TAPS
    if encoder == "resnet50":
        return _BOTTLENECK_TAPS
    raise ValueError(f"unknown segmentor encoder {encoder!r}")


def decoder_widths(decoder_width: int) -> tuple[int, ...]:
    """Return the decoder channel counts for a width knob.

    Args:
        decoder_width: Channel count of the finest decoder stage.

    Returns:
        tuple[int, ...]: Five stages halving from ``decoder_width * 8`` down to
        ``decoder_width``, coarsest first.

    Raises:
        ValueError: If ``decoder_width`` is below one.
    """
    if decoder_width < 1:
        raise ValueError(f"decoder_width must be >= 1; got {decoder_width}")
    return (
        decoder_width * 8,
        decoder_width * 4,
        decoder_width * 2,
        decoder_width,
        decoder_width,
    )


def _construct_encoder(encoder: str, pretrained: bool) -> nn.Module:
    """Return the stock torchvision ResNet used as the encoder."""
    model: nn.Module
    if encoder == "resnet18":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    elif encoder == "resnet34":
        model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
    elif encoder == "resnet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    else:
        raise ValueError(f"unknown segmentor encoder {encoder!r}")
    return model


class ResNetUNet(nn.Module):
    """U-Net decoder over a torchvision ResNet encoder.

    The encoder is kept as its five torchvision blocks so the pretrained
    parameter names survive; only the stem convolution is retargeted to the
    PACT band count. The decoder is a plain chain of upsample-concat-ConvBlock
    stages sized by ``decoder_width``. The forward pass does not apply sigmoid.
    """

    def __init__(
        self,
        encoder: str = "resnet18",
        in_channels: int = 4,
        out_channels: int = 1,
        pretrained: bool = False,
        decoder_width: int = 16,
        separable: bool = False,
    ) -> None:
        """Construct the encoder taps, decoder chain, and logit head.

        Args:
            encoder: Name in ``RESNET_ENCODERS``.
            in_channels: Input band count (flight default 4).
            out_channels: Logit maps (flight default 1).
            pretrained: Load ImageNet encoder weights and remap the stem.
            decoder_width: Channel count of the finest decoder stage.
            separable: Use depthwise-separable decoder convolutions.

        Raises:
            ValueError: If the encoder is not registered or a width is invalid.
        """
        super().__init__()
        backbone = _construct_encoder(encoder, pretrained)
        retarget_first_conv(backbone, in_channels, pretrained)
        self.stem = nn.Sequential(
            backbone.get_submodule("conv1"),
            backbone.get_submodule("bn1"),
            backbone.get_submodule("relu"),
        )
        self.pool = backbone.get_submodule("maxpool")
        self.layer1 = backbone.get_submodule("layer1")
        self.layer2 = backbone.get_submodule("layer2")
        self.layer3 = backbone.get_submodule("layer3")
        self.layer4 = backbone.get_submodule("layer4")

        taps = encoder_tap_channels(encoder)
        widths = decoder_widths(decoder_width)
        decoders: list[ConvBlock] = []
        current = taps[-1]
        # Stages 0..3 consume an encoder skip; the last stage upsamples to the
        # input grid, which has no encoder tap of its own.
        for stage in range(_DECODER_STAGES):
            skip = taps[-2 - stage] if stage < _DECODER_STAGES - 1 else 0
            decoders.append(ConvBlock(current + skip, widths[stage], separable=separable))
            current = widths[stage]
        self.decoders = nn.ModuleList(decoders)
        self.head = nn.Conv2d(current, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a preprocessed batch to a logit mask of the same spatial size.

        Args:
            x: (N, in_channels, H, W) float32 in the normalize_dn domain.

        Returns:
            torch.Tensor: (N, out_channels, H, W) logits.
        """
        s0 = self.stem(x)
        s1 = self.layer1(self.pool(s0))
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        current = self.layer4(s3)
        skips: tuple[torch.Tensor, ...] = (s3, s2, s1, s0)
        for stage, decoder in enumerate(self.decoders):
            target = skips[stage].shape[-2:] if stage < len(skips) else x.shape[-2:]
            current = nn.functional.interpolate(
                current, size=target, mode="bilinear", align_corners=False
            )
            if stage < len(skips):
                current = torch.cat((current, skips[stage]), dim=1)
            current = decoder(current)
        return cast(torch.Tensor, self.head(current))


def build_encoder_segmentor(
    encoder: str,
    in_channels: int = 4,
    out_channels: int = 1,
    pretrained: bool = False,
    decoder_width: int = 16,
    separable: bool = False,
) -> ResNetUNet:
    """Return a ResNet-encoder U-Net segmentor.

    Args:
        encoder: Name in ``RESNET_ENCODERS``.
        in_channels: Input band count.
        out_channels: Logit maps.
        pretrained: Load ImageNet encoder weights.
        decoder_width: Channel count of the finest decoder stage.
        separable: Use depthwise-separable decoder convolutions.

    Returns:
        ResNetUNet: Untrained segmentor in train mode.
    """
    return ResNetUNet(
        encoder=encoder,
        in_channels=in_channels,
        out_channels=out_channels,
        pretrained=pretrained,
        decoder_width=decoder_width,
        separable=separable,
    )
