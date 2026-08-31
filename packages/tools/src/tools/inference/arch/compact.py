"""Compact classifier designed for the plume corpus rather than for ImageNet.

Every other classifier in the registry is a torchvision backbone built to
separate a thousand object categories in natural photographs. This one answers a
single yes-or-no question about a 4-band satellite tile, and it is sized for the
21,350 tiles that are available rather than for the million images those
backbones were designed around. The ResNet-50 baseline reaches a perfect train
score while giving up several points on the held-out split, which is the
signature of a network with more capacity than the corpus supports.

Three choices follow from that:

  - Separable convolutions. A depthwise 3x3 followed by a pointwise 1x1 spends
    roughly a ninth of the parameters of a dense 3x3 at the same width and
    receptive field. The ``full`` modifier restores dense convolutions, so the
    frontier can measure what the separation costs instead of assuming it.
  - Aggressive early downsampling. The stem strides immediately. Plume presence
    is a question about a region, not about a pixel, so the fine resolution that
    a segmentation decoder needs is wasted work here.
  - Global average pooling into a single linear layer. A flattened fully
    connected head would hold more parameters than the entire convolution stack
    and is where a small network of this shape usually overfits first.

Names follow the same grammar as the rest of the registry: ``pactnet`` with an
optional ``w<N>`` stem width (default 16), ``d<N>`` stage count (default 4), and
``full`` for dense convolutions. ``pactnet_w32_d5`` is a wider, deeper variant.

Contains:
  - DEFAULT_COMPACT_WIDTH / DEFAULT_COMPACT_DEPTH: grammar defaults.
  - COMPACT_PREFIX: the name that selects this family.
  - CompactSpec: a parsed point in the compact size space.
  - parse_compact: parse a ``pactnet`` name into a CompactSpec.
  - PactNet: the network itself.
  - build_compact_classifier: construct one from a spec.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from tools.inference.arch.blocks import conv_norm_relu
from tools.inference.arch.grammar import ModifierFlags, parse_modifiers

COMPACT_PREFIX = "pactnet"

DEFAULT_COMPACT_WIDTH = 16
DEFAULT_COMPACT_DEPTH = 4

# Enough to regularise the single linear head without starving a network this
# small of signal.
_HEAD_DROPOUT = 0.2

# Widths double each stage until this ceiling, which keeps the deepest stage of
# a wide variant from holding most of the parameter budget on its own.
_MAX_WIDTH = 256


@dataclass(frozen=True, slots=True)
class CompactSpec:
    """A point in the compact classifier size space.

    Attributes:
        base_width: Channel count emitted by the stem.
        depth: Number of strided stages, including the stem.
        separable: Depthwise-separable convolutions instead of dense ones.
    """

    base_width: int
    depth: int
    separable: bool


def compact_stage_widths(base_width: int, depth: int) -> tuple[int, ...]:
    """Return the channel count of each stage.

    Args:
        base_width: Stem width.
        depth: Stage count including the stem.

    Returns:
        tuple[int, ...]: ``depth`` widths, doubling per stage and held at
        ``256`` once reached.
    """
    widths: list[int] = []
    width = base_width
    for _ in range(depth):
        widths.append(min(width, _MAX_WIDTH))
        width *= 2
    return tuple(widths)


def _conv_block(in_channels: int, out_channels: int, stride: int, separable: bool) -> nn.Module:
    """Return one convolution stage as a batch-normalised, rectified block."""
    return conv_norm_relu(
        in_channels,
        out_channels,
        stride=stride,
        separable=separable,
        mid_norm=separable,
    )


class PactNet(nn.Module):
    """Small separable convolution stack that emits one logit per tile.

    The forward pass maps ``(N, C, H, W)`` to ``(N, 1)``. No sigmoid is applied;
    flight thresholds the logit directly, matching every other classifier in the
    registry.
    """

    def __init__(
        self,
        in_channels: int = 4,
        base_width: int = DEFAULT_COMPACT_WIDTH,
        depth: int = DEFAULT_COMPACT_DEPTH,
        separable: bool = True,
    ) -> None:
        """Build the stack.

        Args:
            in_channels: Input band count.
            base_width: Stem width.
            depth: Strided stage count, including the stem.
            separable: Depthwise-separable convolutions instead of dense ones.

        Raises:
            ValueError: If ``depth`` or ``base_width`` is below one.
        """
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be at least 1, got {depth}")
        if base_width < 1:
            raise ValueError(f"base_width must be at least 1, got {base_width}")
        widths = compact_stage_widths(base_width, depth)
        # The stem is always dense: separating four input bands saves almost
        # nothing and discards the cross-band mixing that the NIR plane exists
        # to provide.
        stages: list[nn.Module] = [_conv_block(in_channels, widths[0], 2, separable=False)]
        for index in range(1, depth):
            stages.append(_conv_block(widths[index - 1], widths[index], 2, separable=separable))
            stages.append(_conv_block(widths[index], widths[index], 1, separable=separable))
        self.features = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(_HEAD_DROPOUT)
        self.head = nn.Linear(widths[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a band stack to one logit per sample.

        Args:
            x: Input of shape ``(N, C, H, W)``.

        Returns:
            torch.Tensor: Logits of shape ``(N, 1)``.
        """
        features = self.features(x)
        pooled = self.pool(features).flatten(1)
        logits: torch.Tensor = self.head(self.dropout(pooled))
        return logits


_COMPACT_FLAGS = ModifierFlags(width=True, depth=True, full=True)


def parse_compact(name: str) -> CompactSpec:
    """Parse a ``pactnet`` architecture name.

    Args:
        name: Grammar name such as ``pactnet``, ``pactnet_w32``, or
            ``pactnet_w8_d5_full``.

    Returns:
        CompactSpec: Parsed width, depth, and convolution style.

    Raises:
        ValueError: If the family or any modifier token is unknown.
    """
    parts = name.split("_")
    if parts[0] != COMPACT_PREFIX:
        raise ValueError(f"unknown compact classifier {name!r}")
    mods = parse_modifiers(tuple(parts[1:]), _COMPACT_FLAGS, "pactnet modifier")
    return CompactSpec(
        base_width=DEFAULT_COMPACT_WIDTH if mods.width is None else mods.width,
        depth=DEFAULT_COMPACT_DEPTH if mods.depth is None else mods.depth,
        separable=True if mods.separable is None else mods.separable,
    )


def build_compact_classifier(spec: CompactSpec, in_channels: int = 4) -> nn.Module:
    """Return an untrained :class:`PactNet` for a parsed spec.

    Args:
        spec: Parsed compact specification.
        in_channels: Input band count (flight default 4).

    Returns:
        nn.Module: Network mapping ``(N, C, H, W)`` to ``(N, 1)`` logits.
    """
    return PactNet(
        in_channels=in_channels,
        base_width=spec.base_width,
        depth=spec.depth,
        separable=spec.separable,
    )
