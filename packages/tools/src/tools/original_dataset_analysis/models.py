"""ShuffleNet classifier and DilateNet segmentor with a fresh stem.

Contains:
  - ShuffleNetClassifier: torchvision ShuffleNet V2 x0.5, untrained stem.
  - DilateNet: dilated segmentor at width 32, four blocks, output stride 4.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import shufflenet_v2_x0_5


def _conv3x3_layers(
    in_channels: int,
    out_channels: int,
    *,
    stride: int,
    dilation: int,
    separable: bool,
) -> list[nn.Module]:
    """Return one 3x3 stage, dense or depthwise-separable."""
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
    return [
        nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        ),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
    ]


def _block(
    in_channels: int,
    out_channels: int,
    stride: int,
    dilation: int,
    *,
    separable: bool,
) -> nn.Sequential:
    """Return one normalized, rectified convolution block."""
    layers = _conv3x3_layers(
        in_channels,
        out_channels,
        stride=stride,
        dilation=dilation,
        separable=separable,
    )
    layers += [nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


class ShuffleNetClassifier(nn.Module):
    """ShuffleNet V2 x0.5 with a randomly initialized first convolution.

    The forward pass maps ``(N, C, H, W)`` to ``(N, 1)`` logits. ImageNet
    weights are not loaded.
    """

    def __init__(self, in_channels: int) -> None:
        """Replace the RGB stem and the 1000-way head.

        Args:
            in_channels: Input band count. Must be at least 1.

        Raises:
            ValueError: If ``in_channels`` is below 1.
        """
        super().__init__()
        if in_channels < 1:
            raise ValueError(f"in_channels must be at least 1, got {in_channels}")
        backbone = shufflenet_v2_x0_5(weights=None)
        first = backbone.conv1[0]
        if not isinstance(first, nn.Conv2d):
            raise TypeError("ShuffleNet conv1[0] is not a Conv2d")
        backbone.conv1[0] = nn.Conv2d(
            in_channels,
            first.out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, 1)
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a band stack to one logit per sample.

        Args:
            x: Input ``(N, C, H, W)``.

        Returns:
            torch.Tensor: Logits ``(N, 1)``.
        """
        logits: torch.Tensor = self.backbone(x)
        return logits


class DilateNet(nn.Module):
    """Dilated segmentor: width 32, four blocks, output stride 4, separable body.

    The forward pass maps ``(N, C, H, W)`` to ``(N, 1, H, W)`` logits.
    """

    def __init__(self, in_channels: int) -> None:
        """Build the dense stem, the separable body, and the logit head.

        Args:
            in_channels: Input band count. Must be at least 1.

        Raises:
            ValueError: If ``in_channels`` is below 1.
        """
        super().__init__()
        if in_channels < 1:
            raise ValueError(f"in_channels must be at least 1, got {in_channels}")
        base_width = 32
        body_width = 64
        stages: list[nn.Module] = [
            _block(in_channels, base_width, 2, 1, separable=False),
            _block(base_width, body_width, 2, 1, separable=True),
        ]
        for rate in (1, 2, 4, 8):
            stages.append(_block(body_width, body_width, 1, rate, separable=True))
        self.features = nn.Sequential(*stages)
        self.head = nn.Conv2d(body_width, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a band stack to a full-resolution logit plane.

        Args:
            x: Input ``(N, C, H, W)``.

        Returns:
            torch.Tensor: Logits ``(N, 1, H, W)``.
        """
        size = x.shape[-2:]
        logits = self.head(self.features(x))
        resized: torch.Tensor = nn.functional.interpolate(
            logits, size=size, mode="bilinear", align_corners=False
        )
        return resized
