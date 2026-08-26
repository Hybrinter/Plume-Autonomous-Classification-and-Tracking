"""Modified ResNet-50 classifier for 4-channel Sentinel-2 input."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


def build_classifier(in_channels: int = 4, pretrained: bool = True) -> nn.Module:
    """Return a ResNet-50 with a 4-channel first conv and a single-logit head."""
    weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet50(weights=weights)
    model.conv1 = nn.Conv2d(
        in_channels, 64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), bias=False
    )
    model.fc = nn.Linear(2048, 1)
    return model
