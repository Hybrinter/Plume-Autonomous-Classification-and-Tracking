"""Dilated fully-convolutional segmentor, an alternative to the U-Net family.

Every other segmentor in the registry is an encoder-decoder: it halves the
spatial size several times, then rebuilds it through a decoder that reads skip
connections from the matching encoder stage. That structure exists to recover
detail the encoder destroyed, and most of its parameters sit in the deepest,
widest stages and in the decoder that mirrors them.

This module takes the other approach. It downsamples twice and then stops,
growing the receptive field with dilated convolutions instead of with further
downsampling. Because the feature map never drops below a quarter of the input,
there is no fine detail to recover and therefore no decoder and no skip
connections to carry. The head is a single 1x1 convolution followed by a
bilinear resize back to the input size.

The trade this makes is worth measuring on this corpus specifically. Plumes are
large, diffuse, and low-contrast rather than small and sharply bounded, so the
detail a decoder is built to restore may not be detail this task needs. A
dilated stack reaches a comparable receptive field for far fewer parameters,
since its blocks all run at one width rather than doubling.

Dilation rates double per block (1, 2, 4, 8, ...). Stacking them this way grows
the receptive field geometrically while each block still reads a 3x3
neighbourhood, which is what lets a shallow stack see a whole plume.

Names follow the registry grammar: ``dilatenet`` with an optional ``w<N>`` stem
width (default 32), ``d<N>`` dilated-block count (default 4), ``s<N>`` output
stride of 4 or 8 (default 4), and ``full`` for dense convolutions in place of
separable ones.

Contains:
  - DILATED_PREFIX: the name that selects this family.
  - DEFAULT_DILATED_WIDTH / DEFAULT_DILATED_BLOCKS / DEFAULT_OUTPUT_STRIDE.
  - VALID_OUTPUT_STRIDES: the supported ``s<N>`` values.
  - DilatedSpec: a parsed point in this size space.
  - dilation_rates: the rate schedule for a block count.
  - DilatedSegmentor: the network itself.
  - parse_dilated: parse a ``dilatenet`` name into a DilatedSpec.
  - build_dilated_segmentor: construct one from a spec.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from tools.inference.arch.blocks import conv_norm_relu
from tools.inference.arch.grammar import ModifierFlags, parse_modifiers

DILATED_PREFIX = "dilatenet"

DEFAULT_DILATED_WIDTH = 32
DEFAULT_DILATED_BLOCKS = 4
DEFAULT_OUTPUT_STRIDE = 4

VALID_OUTPUT_STRIDES: frozenset[int] = frozenset({4, 8})

# The body runs at twice the stem width. One widening step buys most of the
# capacity a doubling ladder would, without the deep, wide stages that make an
# encoder-decoder expensive.
_BODY_MULTIPLIER = 2


@dataclass(frozen=True, slots=True)
class DilatedSpec:
    """A point in the dilated segmentor size space.

    Attributes:
        base_width: Channel count emitted by the stem.
        blocks: Number of dilated blocks in the body.
        output_stride: Spatial reduction held through the body, 4 or 8.
        separable: Depthwise-separable convolutions instead of dense ones.
    """

    base_width: int
    blocks: int
    output_stride: int
    separable: bool


def dilation_rates(blocks: int) -> tuple[int, ...]:
    """Return the dilation rate of each body block.

    Args:
        blocks: Number of dilated blocks.

    Returns:
        tuple[int, ...]: Rates doubling from one, so four blocks give
        ``(1, 2, 4, 8)``.

    Raises:
        ValueError: If ``blocks`` is below one.
    """
    if blocks < 1:
        raise ValueError(f"blocks must be at least 1, got {blocks}")
    return tuple(2**index for index in range(blocks))


def _block(
    in_channels: int,
    out_channels: int,
    stride: int,
    dilation: int,
    separable: bool,
) -> nn.Module:
    """Return one convolution block as a batch-normalised, rectified sequence."""
    return conv_norm_relu(
        in_channels,
        out_channels,
        stride=stride,
        dilation=dilation,
        separable=separable,
        mid_norm=separable,
    )


class DilatedSegmentor(nn.Module):
    """Dilated fully-convolutional segmentor emitting one logit plane.

    The forward pass maps ``(N, C, H, W)`` to ``(N, 1, H, W)``. Spatial size is
    read from the input tensor rather than frozen in the graph, and the output
    is raw logits; flight applies sigmoid after the ONNX session. Both match the
    rest of the segmentor family.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        base_width: int = DEFAULT_DILATED_WIDTH,
        blocks: int = DEFAULT_DILATED_BLOCKS,
        output_stride: int = DEFAULT_OUTPUT_STRIDE,
        separable: bool = True,
    ) -> None:
        """Build the stem, the dilated body, and the logit head.

        Args:
            in_channels: Input band count.
            out_channels: Output plane count.
            base_width: Stem width.
            blocks: Dilated block count.
            output_stride: Spatial reduction held through the body, 4 or 8.
            separable: Depthwise-separable convolutions instead of dense ones.

        Raises:
            ValueError: If ``base_width`` or ``blocks`` is below one, or if
                ``output_stride`` is not in :data:`VALID_OUTPUT_STRIDES`.
        """
        super().__init__()
        if base_width < 1:
            raise ValueError(f"base_width must be at least 1, got {base_width}")
        if output_stride not in VALID_OUTPUT_STRIDES:
            raise ValueError(
                f"output_stride must be one of {sorted(VALID_OUTPUT_STRIDES)}, got {output_stride}"
            )
        rates = dilation_rates(blocks)
        body_width = base_width * _BODY_MULTIPLIER
        # The stem is dense: separating four input bands saves almost nothing
        # and discards the cross-band mixing the NIR plane exists to provide.
        stages: list[nn.Module] = [_block(in_channels, base_width, 2, 1, separable=False)]
        stages.append(_block(base_width, body_width, 2, 1, separable=separable))
        if output_stride == 8:
            stages.append(_block(body_width, body_width, 2, 1, separable=separable))
        for rate in rates:
            stages.append(_block(body_width, body_width, 1, rate, separable=separable))
        self.features = nn.Sequential(*stages)
        self.head = nn.Conv2d(body_width, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a band stack to a full-resolution logit plane.

        Args:
            x: Input of shape ``(N, C, H, W)``.

        Returns:
            torch.Tensor: Logits of shape ``(N, out_channels, H, W)``.
        """
        size = x.shape[-2:]
        logits = self.head(self.features(x))
        resized: torch.Tensor = nn.functional.interpolate(
            logits, size=size, mode="bilinear", align_corners=False
        )
        return resized


_DILATED_FLAGS = ModifierFlags(
    width=True, depth=True, stride=True, full=True, depth_label="block count"
)


def parse_dilated(name: str) -> DilatedSpec:
    """Parse a ``dilatenet`` architecture name.

    Args:
        name: Grammar name such as ``dilatenet``, ``dilatenet_w16``, or
            ``dilatenet_w32_d6_s8_full``.

    Returns:
        DilatedSpec: Parsed width, block count, output stride, and convolution
        style.

    Raises:
        ValueError: If the family or any modifier token is unknown, or if the
            output stride is not supported.
    """
    parts = name.split("_")
    if parts[0] != DILATED_PREFIX:
        raise ValueError(f"unknown dilated segmentor {name!r}")
    mods = parse_modifiers(tuple(parts[1:]), _DILATED_FLAGS, "dilatenet modifier")
    base_width = DEFAULT_DILATED_WIDTH if mods.width is None else mods.width
    blocks = DEFAULT_DILATED_BLOCKS if mods.depth is None else mods.depth
    output_stride = DEFAULT_OUTPUT_STRIDE if mods.stride is None else mods.stride
    separable = True if mods.separable is None else mods.separable
    if output_stride not in VALID_OUTPUT_STRIDES:
        raise ValueError(
            f"output stride must be one of {sorted(VALID_OUTPUT_STRIDES)}, got {output_stride}"
        )
    return DilatedSpec(
        base_width=base_width,
        blocks=blocks,
        output_stride=output_stride,
        separable=separable,
    )


def build_dilated_segmentor(
    spec: DilatedSpec, in_channels: int = 4, out_channels: int = 1
) -> nn.Module:
    """Return an untrained :class:`DilatedSegmentor` for a parsed spec.

    Args:
        spec: Parsed dilated specification.
        in_channels: Input band count (flight default 4).
        out_channels: Output plane count.

    Returns:
        nn.Module: Network mapping ``(N, C, H, W)`` to ``(N, out_channels, H, W)``.
    """
    return DilatedSegmentor(
        in_channels=in_channels,
        out_channels=out_channels,
        base_width=spec.base_width,
        blocks=spec.blocks,
        output_stride=spec.output_stride,
        separable=spec.separable,
    )
