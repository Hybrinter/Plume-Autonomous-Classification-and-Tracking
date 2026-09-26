"""Binary plume classifiers built on torchvision backbones.

Every backbone is retargeted the same way: the stem takes ``in_channels`` bands
and the head emits one logit. The flight default is 3 channels, BLUE, GREEN,
and RED. The graph does not apply sigmoid; flight thresholds the logit directly.

A trailing ``_pt`` on a backbone name loads ImageNet weights and remaps the stem
kernel onto the flight band order (see :mod:`tools.ml_models.arch.stem`). Without
the suffix the network starts from a random initialisation.

ShuffleNet V2 replaces each ``BatchNorm2d`` with a subclass that stays defined
when a channel holds one value. The replacement loads the original batch-norm
state. Other backbones keep stock batch norm.

Contains:
  - BackboneName: registered torchvision backbone names.
  - CLASSIFIER_BACKBONES: string values of BackboneName.
  - BackboneSpec: backbone name split from its pretrained flag.
  - parse_backbone: split a registry name into a BackboneSpec.
  - construct_backbone: stock torchvision network for a spec, still RGB.
  - build_backbone_spec: retarget a spec to PACT bands and one logit.
  - build_backbone: construct one retargeted torchvision classifier by name.
  - build_classifier: construct the default compact pactnet classifier.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import torch
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

from tools.ml_models.arch.stem import retarget_final_linear, retarget_first_conv

PRETRAINED_SUFFIX = "_pt"


class BackboneName(StrEnum):
    """Registered torchvision backbone names."""

    resnet18 = "resnet18"
    resnet34 = "resnet34"
    resnet50 = "resnet50"
    mobilenetv3_small = "mobilenetv3_small"
    mobilenetv3_large = "mobilenetv3_large"
    efficientnet_b0 = "efficientnet_b0"
    shufflenetv2_x0_5 = "shufflenetv2_x0_5"


CLASSIFIER_BACKBONES: frozenset[str] = frozenset(member.value for member in BackboneName)

RESNET_BACKBONES: frozenset[BackboneName] = frozenset(
    {BackboneName.resnet18, BackboneName.resnet34, BackboneName.resnet50}
)


@dataclass(frozen=True, slots=True)
class BackboneSpec:
    """A torchvision backbone choice.

    Attributes:
        backbone: Name in ``BackboneName``.
        pretrained: True when ImageNet weights should be loaded.
    """

    backbone: BackboneName
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
    raw = name[: -len(PRETRAINED_SUFFIX)] if pretrained else name
    try:
        backbone = BackboneName(raw)
    except ValueError:
        raise ValueError(f"unknown classifier backbone {raw!r}") from None
    return BackboneSpec(backbone=backbone, pretrained=pretrained)


def construct_backbone(spec: BackboneSpec) -> nn.Module:
    """Return the stock torchvision network for ``spec``, still RGB and 1000-way.

    Args:
        spec: Parsed backbone name and pretrained flag.

    Returns:
        nn.Module: Unmodified torchvision constructor output.
    """
    load = spec.pretrained
    model: nn.Module
    match spec.backbone:
        case BackboneName.resnet18:
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if load else None)
        case BackboneName.resnet34:
            model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if load else None)
        case BackboneName.resnet50:
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if load else None)
        case BackboneName.mobilenetv3_small:
            model = mobilenet_v3_small(
                weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if load else None
            )
        case BackboneName.mobilenetv3_large:
            model = mobilenet_v3_large(
                weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2 if load else None
            )
        case BackboneName.efficientnet_b0:
            model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if load else None)
        case BackboneName.shufflenetv2_x0_5:
            model = shufflenet_v2_x0_5(
                weights=ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1 if load else None
            )
    return model


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


def build_backbone_spec(spec: BackboneSpec, in_channels: int = 3) -> nn.Module:
    """Return a retargeted torchvision classifier for a parsed spec.

    Args:
        spec: Parsed backbone name and pretrained flag.
        in_channels: Input band count (flight default 3, BLUE/GREEN/RED).

    Returns:
        nn.Module: Network mapping (N, C, H, W) to (N, 1) logits.
    """
    model = construct_backbone(spec)
    retarget_first_conv(model, in_channels, spec.pretrained)
    retarget_final_linear(model, 1)
    if spec.backbone is BackboneName.shufflenetv2_x0_5:
        _use_singleton_safe_batchnorm(model)
    return model


def build_backbone(name: str, in_channels: int = 3) -> nn.Module:
    """Return a retargeted torchvision classifier for a registry name.

    Args:
        name: Registry name such as ``mobilenetv3_small_pt``.
        in_channels: Input band count (flight default 3, BLUE/GREEN/RED).

    Returns:
        nn.Module: Network mapping (N, C, H, W) to (N, 1) logits.

    Raises:
        ValueError: If the backbone is not registered.
    """
    return build_backbone_spec(parse_backbone(name), in_channels=in_channels)


def build_classifier(in_channels: int = 3) -> nn.Module:
    """Return the default compact pactnet classifier.

    Args:
        in_channels: Input band count (flight default 3, BLUE/GREEN/RED).

    Returns:
        nn.Module: Untrained PactNet. Forward maps (N, C, H, W) to (N, 1).

    Notes:
        The compact family is the empty-arch default. Torchvision backbones
        remain available through :func:`build_backbone`.
    """
    from tools.ml_models.arch.compact import (
        DEFAULT_COMPACT_DEPTH,
        DEFAULT_COMPACT_WIDTH,
        CompactSpec,
        build_compact_classifier,
    )

    return build_compact_classifier(
        CompactSpec(
            base_width=DEFAULT_COMPACT_WIDTH,
            depth=DEFAULT_COMPACT_DEPTH,
            separable=True,
        ),
        in_channels=in_channels,
    )
