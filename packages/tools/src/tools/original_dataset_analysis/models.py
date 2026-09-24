"""ShuffleNet classifier and DilateNet segmentor with a fresh stem.

Contains:
  - ShuffleNetClassifier: torchvision ShuffleNet V2 x0.5, untrained stem.
  - DilateNet: dilated segmentor at width 32, four blocks, output stride 4.
  - _SingletonSafeBatchNorm2d: batch norm that accepts one value per channel.
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


class _SingletonSafeBatchNorm2d(nn.BatchNorm2d):
    """Batch norm that stays defined for one value per channel.

    A 30-pixel tile and a batch of one sample reach a 1x1 map inside
    ShuffleNet. Training batch norm rejects that shape. This layer uses the
    running mean and variance for that input.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize one activation map.

        Args:
            x: Input ``(N, C, H, W)``.

        Returns:
            torch.Tensor: Normalized activations with the same shape as ``x``.
        """
        values_per_channel = x.shape[0] * x.shape[2] * x.shape[3]
        if self.training and self.track_running_stats and values_per_channel <= 1:
            normalized: torch.Tensor = nn.functional.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                momentum=0.0,
                eps=self.eps,
            )
            return normalized
        output: torch.Tensor = super().forward(x)
        return output


def _use_singleton_safe_batchnorm(module: nn.Module) -> None:
    """Replace each ``BatchNorm2d`` with :class:`_SingletonSafeBatchNorm2d`.

    Args:
        module: Network edited in place.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d) and type(child) is nn.BatchNorm2d:
            safe = _SingletonSafeBatchNorm2d(
                child.num_features,
                eps=child.eps,
                momentum=child.momentum,
                affine=child.affine,
                track_running_stats=child.track_running_stats,
            )
            safe.load_state_dict(child.state_dict())
            setattr(module, name, safe)
        else:
            _use_singleton_safe_batchnorm(child)


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
        _use_singleton_safe_batchnorm(backbone)
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
