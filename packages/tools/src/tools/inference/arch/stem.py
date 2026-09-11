"""Band-count surgery for ImageNet-pretrained convolution stems.

Torchvision backbones take three RGB planes. PACT feeds four planes ordered
BLUE, GREEN, RED, NIR (`InferenceConfig.input_bands`). Replacing the stem with a
randomly initialised convolution throws away the pretrained first layer and
leaves the downstream batch-norm statistics mismatched, so this module remaps
the pretrained kernel instead.

Two details make the transfer work:

  - Channel order. ImageNet kernels are indexed R, G, B. The PACT permutation
    ``BAND_TO_RGB_INDEX`` sends BLUE to the pretrained blue column, GREEN to
    green, and RED to red rather than aligning them positionally.
  - Response scale. Every extra plane adds another term to the convolution sum,
    which would inflate activations feeding batch-norm layers whose running
    statistics came from three planes. Rescaling by ``3 / in_channels`` keeps
    the expected response where the pretrained statistics expect it.

Contains:
  - BAND_TO_RGB_INDEX: PACT band position to pretrained RGB kernel column.
  - remap_stem_weight: build an ``in_channels`` kernel from an RGB kernel.
  - adapt_conv_in_channels: replace a Conv2d's input band count in place.
  - retarget_first_conv: rewrite a backbone's stem to take ``in_channels``.
  - retarget_final_linear: rewrite a backbone's head to emit one logit.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import torch
from torch import nn

# PACT band order is BLUE, GREEN, RED, NIR; pretrained kernels are R, G, B.
BAND_TO_RGB_INDEX: tuple[int, int, int] = (2, 1, 0)

_RGB_CHANNELS = 3


def remap_stem_weight(weight: torch.Tensor, in_channels: int) -> torch.Tensor:
    """Return an ``in_channels`` stem kernel derived from an RGB kernel.

    Args:
        weight: torch.Tensor[float32, (out, 3, kh, kw)] pretrained kernel.
        in_channels: Target input band count.

    Returns:
        torch.Tensor: (out, in_channels, kh, kw) kernel. The first three bands
        take the pretrained columns under the PACT band permutation; any
        further band takes the mean RGB column, which is the least-committal
        initialisation for a plane with no pretrained counterpart (NIR).
        The whole kernel is scaled by ``3 / in_channels`` so the summed
        response keeps the magnitude the downstream batch-norm statistics were
        fitted against.

    Raises:
        ValueError: If ``weight`` is not a rank-4 RGB kernel or
            ``in_channels`` is below one.
    """
    if weight.ndim != 4 or int(weight.shape[1]) != _RGB_CHANNELS:
        raise ValueError(f"expected an (out, 3, kh, kw) RGB kernel; got {tuple(weight.shape)}")
    if in_channels < 1:
        raise ValueError(f"in_channels must be >= 1; got {in_channels}")
    rgb_mean = weight.mean(dim=1, keepdim=True)
    columns: list[torch.Tensor] = []
    for band in range(in_channels):
        if band < _RGB_CHANNELS:
            columns.append(weight[:, BAND_TO_RGB_INDEX[band] : BAND_TO_RGB_INDEX[band] + 1])
        else:
            columns.append(rgb_mean)
    remapped = torch.cat(columns, dim=1)
    return remapped * (float(_RGB_CHANNELS) / float(in_channels))


def adapt_conv_in_channels(conv: nn.Conv2d, in_channels: int, pretrained: bool) -> nn.Conv2d:
    """Return a copy of ``conv`` that accepts ``in_channels`` planes.

    Args:
        conv: Source stem convolution, normally RGB.
        in_channels: Target input band count.
        pretrained: When true, seed the new kernel from ``conv.weight`` via
            :func:`remap_stem_weight`. When false, leave the fresh random
            initialisation alone.

    Returns:
        nn.Conv2d: New convolution with the source geometry (kernel size,
        stride, padding, dilation, groups, bias) and the requested band count.
        The source is returned unchanged when it already matches.
    """
    if int(conv.in_channels) == int(in_channels):
        return conv
    replacement = nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,  # type: ignore[arg-type]
        stride=conv.stride,  # type: ignore[arg-type]
        padding=conv.padding,  # type: ignore[arg-type]
        dilation=conv.dilation,  # type: ignore[arg-type]
        groups=conv.groups,
        bias=conv.bias is not None,
    )
    if pretrained and int(conv.in_channels) == _RGB_CHANNELS:
        with torch.no_grad():
            replacement.weight.copy_(remap_stem_weight(conv.weight.detach(), in_channels))
            if conv.bias is not None and replacement.bias is not None:
                replacement.bias.copy_(conv.bias.detach())
    return replacement


def _attach(root: nn.Module, qualified_name: str, replacement: nn.Module) -> None:
    """Bind ``replacement`` at ``qualified_name`` under ``root``.

    Args:
        root: Module that owns the target.
        qualified_name: Dotted path as reported by ``named_modules``.
        replacement: Module to bind.
    """
    parent_name, _, leaf = qualified_name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, leaf, replacement)


def retarget_first_conv(model: nn.Module, in_channels: int, pretrained: bool) -> None:
    """Rewrite ``model``'s stem convolution to accept ``in_channels`` planes.

    Args:
        model: Backbone whose first convolution is the stem.
        in_channels: Target input band count.
        pretrained: Forwarded to :func:`adapt_conv_in_channels`.

    Raises:
        ValueError: If the model contains no convolution.

    Notes:
        Locating the stem by traversal rather than by a per-family index keeps
        one code path across the ResNet, MobileNet, EfficientNet, and ShuffleNet
        families, whose stems sit at different attribute paths.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            _attach(model, name, adapt_conv_in_channels(module, in_channels, pretrained))
            return
    raise ValueError("model has no Conv2d stem to retarget")


def retarget_final_linear(model: nn.Module, out_features: int) -> None:
    """Rewrite ``model``'s classification head to emit ``out_features`` logits.

    Args:
        model: Backbone whose last linear layer is the head.
        out_features: Logit count. PACT uses one for binary plume presence.

    Raises:
        ValueError: If the model contains no linear layer.
    """
    target: tuple[str, nn.Linear] | None = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            target = (name, module)
    if target is None:
        raise ValueError("model has no Linear head to retarget")
    name, head = target
    _attach(model, name, nn.Linear(head.in_features, out_features))
