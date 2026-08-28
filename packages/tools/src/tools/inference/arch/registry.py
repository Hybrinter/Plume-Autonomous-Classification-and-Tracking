"""Architecture registry: kind plus name to a builder.

Contains:
  - DEFAULT_ARCH: kind to default architecture name.
  - default_arch: resolve an empty name.
  - build: construct an untrained network for a kind and name.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

from torch import nn

from tools.inference.arch.classifier import build_classifier
from tools.inference.arch.unet import build_segmentor

DEFAULT_ARCH: dict[str, str] = {
    "classifier": "resnet50",
    "segmentor": "unet",
}

_KNOWN: frozenset[tuple[str, str]] = frozenset({("classifier", "resnet50"), ("segmentor", "unet")})


def default_arch(kind: str) -> str:
    """Return the default architecture name for a train kind.

    Args:
        kind: ``classifier`` or ``segmentor``.

    Returns:
        str: ``resnet50`` or ``unet``.

    Raises:
        ValueError: If kind is unknown.
    """
    if kind not in DEFAULT_ARCH:
        raise ValueError(f"unknown train kind {kind!r}")
    return DEFAULT_ARCH[kind]


def resolve_arch(kind: str, arch: str) -> str:
    """Return ``arch`` or the default for ``kind``.

    Args:
        kind: ``classifier`` or ``segmentor``.
        arch: Architecture name. Empty string selects the default.

    Returns:
        str: Resolved name.

    Raises:
        ValueError: If the pair is unknown.
    """
    name = arch if arch else default_arch(kind)
    if (kind, name) not in _KNOWN:
        raise ValueError(f"unknown architecture {kind}/{name}")
    return name


def build(kind: str, arch: str, in_channels: int) -> nn.Module:
    """Construct the untrained network for ``kind`` and ``arch``.

    Args:
        kind: ``classifier`` or ``segmentor``.
        arch: Architecture name. Empty string selects the default.
        in_channels: Input band count.

    Returns:
        nn.Module: Untrained graph that emits logits.

    Raises:
        ValueError: If the pair is unknown.
    """
    name = resolve_arch(kind, arch)
    if kind == "classifier" and name == "resnet50":
        return build_classifier(in_channels=in_channels)
    if kind == "segmentor" and name == "unet":
        return build_segmentor(in_channels=in_channels, out_channels=1)
    raise ValueError(f"unknown architecture {kind}/{name}")
