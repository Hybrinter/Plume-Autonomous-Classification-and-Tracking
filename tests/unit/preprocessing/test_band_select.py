"""Unit tests for pact.preprocessing.band_select — select_bands() and BAND_INDICES.

Satisfies: §6.2 of PACT_SW_ARCH.md — Preprocessing subsystem unit tests.
REQ-AIML-PREP-001, REQ-AIML-IMAG-001
"""

from __future__ import annotations

# third-party
import numpy as np
import pytest

# module under test
from pact.preprocessing.band_select import BAND_INDICES, select_bands


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_raw_bands(n_bands: int = 4, h: int = 8, w: int = 8) -> np.ndarray:
    """Return a (n_bands, H, W) float32 array where band i has all pixels == float(i)."""
    arr = np.zeros((n_bands, h, w), dtype=np.float32)
    for i in range(n_bands):
        arr[i, :, :] = float(i)
    return arr  # np.ndarray[float32, (n_bands, H, W)]


# ---------------------------------------------------------------------------
# BAND_INDICES constant tests
# ---------------------------------------------------------------------------


def test_band_indices_constant() -> None:
    """BAND_INDICES must contain entries for B2, B3, B4, and B8."""
    for band in ("B2", "B3", "B4", "B8"):
        assert band in BAND_INDICES, f"BAND_INDICES missing key '{band}'"


def test_band_indices_values_unique() -> None:
    """All BAND_INDICES values must be unique (no two band names map to the same index)."""
    values = list(BAND_INDICES.values())
    assert len(values) == len(set(values)), "BAND_INDICES contains duplicate index values"


def test_band_indices_values_in_range() -> None:
    """BAND_INDICES values must be valid indices for a 4-channel input array (0–3)."""
    for name, idx in BAND_INDICES.items():
        assert 0 <= idx <= 3, f"BAND_INDICES['{name}'] = {idx} is out of range [0, 3]"


# ---------------------------------------------------------------------------
# select_bands() tests
# ---------------------------------------------------------------------------


def test_select_single_band() -> None:
    """select_bands with ('B2',) must return shape (1, H, W) with the B2 channel's values."""
    raw = _make_raw_bands()  # band 0 → all zeros (B2), band 1 → all ones (B3), etc.
    result = select_bands(raw, band_names=("B2",))
    assert result.shape == (1, 8, 8), f"Expected (1,8,8), got {result.shape}"
    expected_value = float(BAND_INDICES["B2"])
    np.testing.assert_array_almost_equal(result[0], np.full((8, 8), expected_value))


def test_select_all_bands() -> None:
    """select_bands with all four band names returns shape (4, H, W)."""
    raw = _make_raw_bands()
    result = select_bands(raw, band_names=("B2", "B3", "B4", "B8"))
    assert result.shape == (4, 8, 8), f"Expected (4,8,8), got {result.shape}"


def test_select_bands_order_preserved() -> None:
    """select_bands must return channels in the requested order, not in BAND_INDICES order."""
    raw = _make_raw_bands()
    # Request B8 first, then B2 — output channel 0 should be B8's value
    result = select_bands(raw, band_names=("B8", "B2"))
    b8_value = float(BAND_INDICES["B8"])
    b2_value = float(BAND_INDICES["B2"])
    np.testing.assert_array_almost_equal(result[0], np.full((8, 8), b8_value))
    np.testing.assert_array_almost_equal(result[1], np.full((8, 8), b2_value))


def test_select_bands_dtype_preserved() -> None:
    """select_bands must return float32 output when given float32 input."""
    raw = _make_raw_bands()
    result = select_bands(raw, band_names=("B2", "B3", "B4", "B8"))
    assert result.dtype == np.float32, f"Expected float32 output, got {result.dtype}"
