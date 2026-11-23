"""Shared architecture-name modifier grammar.

Registry families share one underscore-separated token language. Width, depth,
stride, and decoder-width tokens are ``<prefix><digits>``. Flag tokens select
convolution style and pretrained weights. Each family names the tokens it
accepts; an unknown token still raises with that family's error text.

Contains:
  - ModifierFlags: which tokens a family accepts, plus numeric-token labels.
  - NameModifiers: parsed numeric values and flags, unset fields left as None.
  - positive_int: parse a ``<prefix><digits>`` token into a positive int.
  - parse_modifiers: walk tokens under one family's flags.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModifierFlags:
    """Which modifier tokens one architecture family accepts.

    Attributes:
        width: Accept ``w<N>`` stem-width tokens.
        depth: Accept ``d<N>`` depth or block-count tokens.
        stride: Accept ``s<N>`` output-stride tokens.
        decoder_width: Accept ``x<N>`` decoder-width tokens.
        sep: Accept ``sep`` (depthwise-separable convolutions).
        full: Accept ``full`` (dense convolutions).
        pt: Accept ``pt`` (ImageNet encoder weights).
        depth_label: Error-text label for ``d<N>`` tokens.
        stride_label: Error-text label for ``s<N>`` tokens.
        decoder_width_label: Error-text label for ``x<N>`` tokens.
    """

    width: bool = False
    depth: bool = False
    stride: bool = False
    decoder_width: bool = False
    sep: bool = False
    full: bool = False
    pt: bool = False
    depth_label: str = "depth"
    stride_label: str = "output stride"
    decoder_width_label: str = "decoder width"


@dataclass(frozen=True, slots=True)
class NameModifiers:
    """Parsed modifier values. ``None`` means the token was absent.

    Attributes:
        width: Stem width from ``w<N>``.
        depth: Stage or block count from ``d<N>``.
        stride: Output stride from ``s<N>``.
        decoder_width: Decoder width from ``x<N>``.
        separable: ``True`` after ``sep``, ``False`` after ``full``, else None.
        pretrained: ``True`` after ``pt``.
    """

    width: int | None = None
    depth: int | None = None
    stride: int | None = None
    decoder_width: int | None = None
    separable: bool | None = None
    pretrained: bool = False


def positive_int(token: str, prefix: str, label: str) -> int:
    """Parse a ``<prefix><digits>`` token into a positive int.

    Args:
        token: Full token such as ``w16``.
        prefix: Leading characters already matched, such as ``w``.
        label: Noun used in the malformed-token error, such as ``width``.

    Returns:
        int: Parsed value, at least 1.

    Raises:
        ValueError: If the remainder is empty, non-numeric, or below one.

    Notes:
        Prefix matching is positional. ``wide`` is a malformed ``w`` token, not
        an unknown modifier, when width tokens are in the family's flags.
    """
    digits = token[len(prefix) :]
    if not digits.isdigit() or int(digits) < 1:
        raise ValueError(f"malformed {label} token {token!r} in architecture name")
    return int(digits)


def parse_modifiers(
    tokens: tuple[str, ...],
    flags: ModifierFlags,
    unknown: str,
) -> NameModifiers:
    """Walk modifier tokens under one family's accepted flags.

    Args:
        tokens: Underscore-split tokens after the family prefix.
        flags: Accepted tokens and numeric-token labels.
        unknown: Phrase in the unknown-token error, such as ``pactnet modifier``.

    Returns:
        NameModifiers: Parsed fields. Absent tokens stay at their defaults.

    Raises:
        ValueError: If a token is not in ``flags``, or a numeric token is
            malformed.

    Notes:
        ``sep`` and ``full`` both write ``separable``. Families enable at most
        one of those flags. Later tokens overwrite earlier ones of the same
        kind.
    """
    width: int | None = None
    depth: int | None = None
    stride: int | None = None
    decoder_width: int | None = None
    separable: bool | None = None
    pretrained = False
    for token in tokens:
        match token:
            case "full" if flags.full:
                separable = False
            case "sep" if flags.sep:
                separable = True
            case "pt" if flags.pt:
                pretrained = True
            case _ if flags.width and token.startswith("w"):
                width = positive_int(token, "w", "width")
            case _ if flags.depth and token.startswith("d"):
                depth = positive_int(token, "d", flags.depth_label)
            case _ if flags.stride and token.startswith("s"):
                stride = positive_int(token, "s", flags.stride_label)
            case _ if flags.decoder_width and token.startswith("x"):
                decoder_width = positive_int(token, "x", flags.decoder_width_label)
            case _:
                raise ValueError(f"unknown {unknown} {token!r}")
    return NameModifiers(
        width=width,
        depth=depth,
        stride=stride,
        decoder_width=decoder_width,
        separable=separable,
        pretrained=pretrained,
    )
