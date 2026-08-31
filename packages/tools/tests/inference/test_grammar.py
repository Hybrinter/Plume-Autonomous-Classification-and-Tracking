"""Shared architecture-name grammar tests."""

import pytest
from tools.inference.arch.grammar import (
    ModifierFlags,
    NameModifiers,
    parse_modifiers,
    positive_int,
)


def test_positive_int_parses_digits() -> None:
    """A prefix plus digits returns the integer value."""
    assert positive_int("w16", "w", "width") == 16
    assert positive_int("d3", "d", "depth") == 3


@pytest.mark.parametrize("token", ["w", "w0", "wide"])
def test_positive_int_rejects_malformed(token: str) -> None:
    """Empty, zero, or non-numeric remainders raise ValueError."""
    with pytest.raises(ValueError, match="malformed width"):
        positive_int(token, "w", "width")


def test_parse_modifiers_reads_enabled_tokens() -> None:
    """Enabled numeric and flag tokens fill NameModifiers."""
    flags = ModifierFlags(width=True, depth=True, full=True)
    mods = parse_modifiers(("w8", "d5", "full"), flags, "pactnet modifier")
    assert mods == NameModifiers(width=8, depth=5, separable=False, pretrained=False)


def test_parse_modifiers_keeps_absent_fields_none() -> None:
    """Tokens that are not present stay at their defaults."""
    flags = ModifierFlags(width=True, sep=True, pt=True)
    mods = parse_modifiers(("sep", "pt"), flags, "runet modifier")
    assert mods.width is None
    assert mods.separable is True
    assert mods.pretrained is True


def test_parse_modifiers_rejects_disabled_tokens() -> None:
    """A token outside the family's flags uses that family's error phrase."""
    flags = ModifierFlags(width=True)
    with pytest.raises(ValueError, match="unknown pactnet modifier"):
        parse_modifiers(("sep",), flags, "pactnet modifier")
