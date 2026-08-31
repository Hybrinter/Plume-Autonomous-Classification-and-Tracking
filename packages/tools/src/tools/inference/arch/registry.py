"""Architecture registry: kind plus name to a builder.

Architecture names are a small grammar rather than a fixed enumeration, so a
sweep can name a point in the size/quality space directly in its TOML without
a code change per variant.

Classifier names come from two families:

  - A torchvision backbone with an optional ``_pt`` suffix for ImageNet weights,
    for example ``resnet18``, ``mobilenetv3_small_pt``.
  - ``pactnet``, the compact stack built for this corpus. ``w<N>`` sets the stem
    width (default 16), ``d<N>`` the stage count (default 4), and ``full``
    selects dense convolutions over separable ones.

Segmentor names come from three families:

  - ``unet`` is the scratch U-Net. ``w<N>`` sets the stem width (default 64),
    ``d<N>`` sets the stage count (default 4), and ``sep`` selects
    depthwise-separable convolutions. ``unet_w16_d3_sep`` is a narrow, shallow,
    separable variant; a bare ``unet`` is the original baseline.
  - ``runet18`` / ``runet34`` / ``runet50`` put a U-Net decoder on a ResNet
    encoder. ``pt`` loads ImageNet encoder weights, ``x<N>`` sets the decoder
    width (default 16), and ``sep`` applies to the decoder.
  - ``dilatenet`` drops the decoder entirely and grows the receptive field with
    dilated convolutions. ``w<N>`` sets the stem width (default 32), ``d<N>``
    the dilated-block count (default 4), ``s<N>`` the output stride (4 or 8),
    and ``full`` selects dense convolutions.

``known()`` returns a curated catalog spanning that space for listings and
tests; ``resolve_arch`` accepts any well-formed name.

Contains:
  - DEFAULT_ARCH: kind to default architecture name.
  - UNetSpec / EncoderUNetSpec / DilatedSpec: parsed segmentor names.
  - parse_classifier: parse a classifier name into one of its two specs.
  - parse_segmentor: parse a segmentor name into one of those specs.
  - default_arch: resolve an empty name.
  - known: curated catalog of kind/name pairs.
  - resolve_arch: validate a name against the grammar.
  - build: construct an untrained network for a kind and name.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from tools.inference.arch.classifier import BackboneSpec, build_backbone_spec, parse_backbone
from tools.inference.arch.compact import (
    COMPACT_PREFIX,
    CompactSpec,
    build_compact_classifier,
    parse_compact,
)
from tools.inference.arch.dilated import (
    DILATED_PREFIX,
    DilatedSpec,
    build_dilated_segmentor,
    parse_dilated,
)
from tools.inference.arch.encoder_unet import RESNET_ENCODERS, build_encoder_segmentor
from tools.inference.arch.grammar import ModifierFlags, parse_modifiers
from tools.inference.arch.unet import build_segmentor

DEFAULT_ARCH: dict[str, str] = {
    "classifier": "pactnet",
    "segmentor": "dilatenet",
}

DEFAULT_BASE_WIDTH = 64
DEFAULT_DEPTH = 4
DEFAULT_DECODER_WIDTH = 16

_ENCODER_PREFIX = "runet"

_UNET_FLAGS = ModifierFlags(width=True, depth=True, sep=True)
_RUNET_FLAGS = ModifierFlags(decoder_width=True, sep=True, pt=True)

# Curated points across the size/quality space. Every entry is also a valid
# grammar name; the catalog exists so `arches` prints something useful and so
# tests have a stable list to walk.
_CATALOG: frozenset[tuple[str, str]] = frozenset(
    {
        ("classifier", "resnet50"),
        ("classifier", "resnet50_pt"),
        ("classifier", "resnet34"),
        ("classifier", "resnet34_pt"),
        ("classifier", "resnet18"),
        ("classifier", "resnet18_pt"),
        ("classifier", "efficientnet_b0"),
        ("classifier", "efficientnet_b0_pt"),
        ("classifier", "mobilenetv3_large"),
        ("classifier", "mobilenetv3_large_pt"),
        ("classifier", "mobilenetv3_small"),
        ("classifier", "mobilenetv3_small_pt"),
        ("classifier", "shufflenetv2_x0_5"),
        ("classifier", "shufflenetv2_x0_5_pt"),
        ("classifier", "pactnet"),
        ("classifier", "pactnet_w8"),
        ("classifier", "pactnet_w32"),
        ("classifier", "pactnet_w16_d5"),
        ("classifier", "pactnet_w16_full"),
        ("segmentor", "unet"),
        ("segmentor", "unet_w32"),
        ("segmentor", "unet_w16"),
        ("segmentor", "unet_w8"),
        ("segmentor", "unet_w32_sep"),
        ("segmentor", "unet_w16_sep"),
        ("segmentor", "unet_w32_d3"),
        ("segmentor", "unet_w16_d3"),
        ("segmentor", "runet18"),
        ("segmentor", "runet18_pt"),
        ("segmentor", "runet18_pt_x8"),
        ("segmentor", "runet18_pt_x32"),
        ("segmentor", "runet34_pt"),
        ("segmentor", "runet50_pt"),
        ("segmentor", "dilatenet"),
        ("segmentor", "dilatenet_w16"),
        ("segmentor", "dilatenet_w64"),
        ("segmentor", "dilatenet_w32_d6"),
        ("segmentor", "dilatenet_w32_s8"),
    }
)


@dataclass(frozen=True, slots=True)
class UNetSpec:
    """A scratch U-Net point in the size/quality space.

    Attributes:
        base_width: Stem channel count.
        depth: Stage count including the stem.
        separable: Depthwise-separable convolutions.
    """

    base_width: int
    depth: int
    separable: bool


@dataclass(frozen=True, slots=True)
class EncoderUNetSpec:
    """A ResNet-encoder U-Net point in the size/quality space.

    Attributes:
        encoder: Name in ``RESNET_ENCODERS``.
        pretrained: Load ImageNet encoder weights.
        decoder_width: Channel count of the finest decoder stage.
        separable: Depthwise-separable decoder convolutions.
    """

    encoder: str
    pretrained: bool
    decoder_width: int
    separable: bool


def _parse_scratch_unet(tokens: tuple[str, ...]) -> UNetSpec:
    """Parse the modifier tokens of a ``unet`` name."""
    mods = parse_modifiers(tokens, _UNET_FLAGS, "unet modifier")
    return UNetSpec(
        base_width=DEFAULT_BASE_WIDTH if mods.width is None else mods.width,
        depth=DEFAULT_DEPTH if mods.depth is None else mods.depth,
        separable=False if mods.separable is None else mods.separable,
    )


def _parse_encoder_unet(head: str, tokens: tuple[str, ...]) -> EncoderUNetSpec:
    """Parse a ``runet<encoder>`` name and its modifier tokens."""
    encoder = f"resnet{head[len(_ENCODER_PREFIX) :]}"
    if encoder not in RESNET_ENCODERS:
        raise ValueError(f"unknown segmentor encoder {encoder!r}")
    mods = parse_modifiers(tokens, _RUNET_FLAGS, "runet modifier")
    return EncoderUNetSpec(
        encoder=encoder,
        pretrained=mods.pretrained,
        decoder_width=(DEFAULT_DECODER_WIDTH if mods.decoder_width is None else mods.decoder_width),
        separable=False if mods.separable is None else mods.separable,
    )


def parse_segmentor(name: str) -> UNetSpec | EncoderUNetSpec | DilatedSpec:
    """Parse a segmentor architecture name.

    Args:
        name: Grammar name such as ``unet_w16_sep``, ``runet18_pt_x32``, or
            ``dilatenet_w32_d6``.

    Returns:
        UNetSpec | EncoderUNetSpec | DilatedSpec: Parsed family and modifiers.

    Raises:
        ValueError: If the family or any modifier token is unknown.
    """
    parts = name.split("_")
    head = parts[0]
    tokens = tuple(parts[1:])
    match head:
        case "unet":
            return _parse_scratch_unet(tokens)
        case _ if head == DILATED_PREFIX:
            return parse_dilated(name)
        case _ if head.startswith(_ENCODER_PREFIX):
            return _parse_encoder_unet(head, tokens)
        case _:
            raise ValueError(f"unknown segmentor architecture {name!r}")


def parse_classifier(name: str) -> BackboneSpec | CompactSpec:
    """Parse a classifier architecture name.

    Args:
        name: Grammar name such as ``resnet18_pt`` or ``pactnet_w32_d5``.

    Returns:
        BackboneSpec | CompactSpec: A torchvision backbone choice, or a point in
        the compact family.

    Raises:
        ValueError: If the family or any modifier token is unknown.
    """
    head = name.split("_")[0]
    match head:
        case _ if head == COMPACT_PREFIX:
            return parse_compact(name)
        case _:
            return parse_backbone(name)


def known() -> frozenset[tuple[str, str]]:
    """Return the curated catalog of ``(kind, name)`` pairs.

    Returns:
        frozenset[tuple[str, str]]: Points spanning the classifier backbones
        and both segmentor families. Names outside this catalog are still
        valid when they parse; see :func:`resolve_arch`.
    """
    return _CATALOG


def default_arch(kind: str) -> str:
    """Return the default architecture name for a train kind.

    Args:
        kind: ``classifier`` or ``segmentor``.

    Returns:
        str: ``pactnet`` or ``dilatenet``.

    Raises:
        ValueError: If kind is unknown.
    """
    if kind not in DEFAULT_ARCH:
        raise ValueError(f"unknown train kind {kind!r}")
    return DEFAULT_ARCH[kind]


def resolve_arch(kind: str, arch: str) -> str:
    """Return ``arch`` or the default for ``kind``, after validating it.

    Args:
        kind: ``classifier`` or ``segmentor``.
        arch: Architecture name. Empty string selects the default.

    Returns:
        str: Resolved name.

    Raises:
        ValueError: If the kind is unknown or the name does not parse.
    """
    name = arch if arch else default_arch(kind)
    match kind:
        case "classifier":
            parse_classifier(name)
            return name
        case "segmentor":
            parse_segmentor(name)
            return name
        case _:
            raise ValueError(f"unknown train kind {kind!r}")


def build(kind: str, arch: str, in_channels: int) -> nn.Module:
    """Construct the untrained network for ``kind`` and ``arch``.

    Args:
        kind: ``classifier`` or ``segmentor``.
        arch: Architecture name. Empty string selects the default.
        in_channels: Input band count.

    Returns:
        nn.Module: Untrained graph that emits logits.

    Raises:
        ValueError: If the kind is unknown or the name does not parse.
    """
    name = resolve_arch(kind, arch)
    match kind:
        case "classifier":
            match parse_classifier(name):
                case CompactSpec() as compact:
                    return build_compact_classifier(compact, in_channels=in_channels)
                case BackboneSpec() as backbone:
                    return build_backbone_spec(backbone, in_channels=in_channels)
        case "segmentor":
            match parse_segmentor(name):
                case DilatedSpec() as dilated:
                    return build_dilated_segmentor(dilated, in_channels=in_channels, out_channels=1)
                case UNetSpec() as unet:
                    return build_segmentor(
                        in_channels=in_channels,
                        out_channels=1,
                        base_width=unet.base_width,
                        depth=unet.depth,
                        separable=unet.separable,
                    )
                case EncoderUNetSpec() as encoder:
                    return build_encoder_segmentor(
                        encoder.encoder,
                        in_channels=in_channels,
                        out_channels=1,
                        pretrained=encoder.pretrained,
                        decoder_width=encoder.decoder_width,
                        separable=encoder.separable,
                    )
        case _:
            raise ValueError(f"unknown train kind {kind!r}")
