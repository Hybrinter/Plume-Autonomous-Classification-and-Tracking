"""Binary plume classifiers built on torchvision backbones.

Every backbone is retargeted the same way: the stem takes `in_channels` bands
(flight default 4) instead of three RGB planes, and the head emits one logit.
The graph does not apply sigmoid; flight thresholds the logit directly.

A trailing ``_pt`` on a backbone name loads ImageNet weights and remaps the stem
kernel onto the PACT band order (see :mod:`tools.inference.arch.stem`). Without
the suffix the network starts from a random initialisation, which is what the
original ResNet-50 baseline did.

Contains:
  - CLASSIFIER_BACKBONES: registered backbone names, pretrained variants aside.
  - BackboneSpec: backbone name split from its pretrained flag.
  - parse_backbone: split a registry name into a BackboneSpec.
  - build_backbone: construct one retargeted torchvision classifier.
  - build_classifier: construct the default ResNet-50 classifier.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Large_Weights,
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ShuffleNet_V2_X0_5_Weights,
    efficientnet_b0,
    mobilenet_v3_large,
    mobilenet_v3_small,
    resnet18,
    resnet34,
    resnet50,
    shufflenet_v2_x0_5,
)

from tools.inference.arch.stem import retarget_final_linear, retarget_first_conv

PRETRAINED_SUFFIX = "_pt"

CLASSIFIER_BACKBONES: frozenset[str] = frozenset(
    {
        "resnet18",
        "resnet34",
        "resnet50",
        "mobilenetv3_small",
        "mobilenetv3_large",
        "efficientnet_b0",
        "shufflenetv2_x0_5",
    }
)


@dataclass(frozen=True, slots=True)
class BackboneSpec:
    """A torchvision backbone choice.

    Attributes:
        backbone: Name in ``CLASSIFIER_BACKBONES``.
        pretrained: True when ImageNet weights should be loaded.
    """

    backbone: str
    pretrained: bool


def parse_backbone(name: str) -> BackboneSpec:
    """Split a registry architecture name into backbone and pretrained flag.

    Args:
        name: Registry name such as ``resnet18`` or ``resnet18_pt``.

    Returns:
        BackboneSpec: Backbone name with the ``_pt`` suffix removed and the
        pretrained flag set accordingly.

    Raises:
        ValueError: If the backbone is not registered.
    """
    pretrained = name.endswith(PRETRAINED_SUFFIX)
    backbone = name[: -len(PRETRAINED_SUFFIX)] if pretrained else name
    if backbone not in CLASSIFIER_BACKBONES:
        raise ValueError(f"unknown classifier backbone {backbone!r}")
    return BackboneSpec(backbone=backbone, pretrained=pretrained)


def _construct(spec: BackboneSpec) -> nn.Module:
    """Return the stock torchvision network for ``spec``, still RGB and 1000-way."""
    load = spec.pretrained
    model: nn.Module
    if spec.backbone == "resnet18":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if load else None)
    elif spec.backbone == "resnet34":
        model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if load else None)
    elif spec.backbone == "resnet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if load else None)
    elif spec.backbone == "mobilenetv3_small":
        model = mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if load else None
        )
    elif spec.backbone == "mobilenetv3_large":
        model = mobilenet_v3_large(
            weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2 if load else None
        )
    elif spec.backbone == "efficientnet_b0":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if load else None)
    elif spec.backbone == "shufflenetv2_x0_5":
        model = shufflenet_v2_x0_5(
            weights=ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1 if load else None
        )
    else:
        raise ValueError(f"unknown classifier backbone {spec.backbone!r}")
    return model


def build_backbone(name: str, in_channels: int = 4) -> nn.Module:
    """Return a retargeted torchvision classifier for a registry name.

    Args:
        name: Registry name such as ``mobilenetv3_small_pt``.
        in_channels: Input band count (flight default 4).

    Returns:
        nn.Module: Network mapping (N, C, H, W) to (N, 1) logits.

    Raises:
        ValueError: If the backbone is not registered.
    """
    spec = parse_backbone(name)
    model = _construct(spec)
    retarget_first_conv(model, in_channels, spec.pretrained)
    retarget_final_linear(model, 1)
    return model


def build_classifier(in_channels: int = 4) -> nn.Module:
    """Return the default ResNet-50 classifier with random weights.

    Args:
        in_channels: Input band count (flight default 4).

    Returns:
        nn.Module: Untrained ResNet-50. Forward maps (N, C, H, W) to (N, 1).

    Notes:
        This is the historical baseline: torchvision geometry, no ImageNet
        checkpoint. ``build_backbone("resnet50_pt", ...)`` is the pretrained
        counterpart.
    """
    return build_backbone("resnet50", in_channels=in_channels)
