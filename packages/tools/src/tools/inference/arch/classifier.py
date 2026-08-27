"""ResNet-50 binary plume classifier with a 4-channel stem.

Uses torchvision ResNet-50. The first convolution accepts `in_channels` bands
(flight default 4). The final linear layer emits one logit. Weights start
random; ImageNet checkpoints are not loaded. The graph does not apply sigmoid.

Contains:
  - build_classifier: construct the 4-channel, 1-logit ResNet-50.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

from typing import cast

from torch import nn
from torchvision.models import resnet50


def build_classifier(in_channels: int = 4) -> nn.Module:
    """Return a ResNet-50 with an `in_channels` stem and a single logit.

    Args:
        in_channels: Input band count (flight default 4).

    Returns:
        nn.Module: Untrained ResNet-50. Forward maps (N, C, H, W) to (N, 1).

    Notes:
        The stem keeps the torchvision 7x7 / stride-2 / pad-3 geometry and only
        changes the incoming channel count. `weights=None` avoids a pretrained
        download.
    """
    model = resnet50(weights=None)
    stem = model.conv1
    model.conv1 = nn.Conv2d(
        in_channels,
        stem.out_channels,
        kernel_size=stem.kernel_size,
        stride=stem.stride,
        padding=stem.padding,
        bias=False,
    )
    model.fc = nn.Linear(model.fc.in_features, 1)
    return cast(nn.Module, model)
