"""Band-order and subset selection."""

from __future__ import annotations

import pytest
from tools.original_dataset_analysis.bands import (
    BandSpec,
    coerce_descriptions,
    resolve_subset,
    verify_band_order,
)

_HYPOTHESIS = (
    "B1 coastal",
    "B2 blue",
    "B3 green",
    "B4 red",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B9",
    "B11",
    "B12",
    "B10 cirrus",
)

_STANDARD = (
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B10",
    "B11",
    "B12",
)


def test_hypothesis_order_drops_b10_by_name() -> None:
    """The 12-band set and RGB follow descriptions, including B10 at the last index."""
    order = verify_band_order(_HYPOTHESIS)
    rgb = resolve_subset(order, BandSpec("rgb"))
    full = resolve_subset(order, BandSpec("s2_12"))
    assert rgb.indices == (1, 2, 3)
    assert rgb.ids == ("B2", "B3", "B4")
    assert full.name == "s2_12"
    assert "B10" not in full.ids
    assert full.indices == tuple(index for index in range(13) if index != 12)


def test_standard_slot_still_drops_b10() -> None:
    """Moving B10 to index 10 still removes B10 and keeps B2, B3, and B4."""
    order = verify_band_order(_STANDARD)
    assert order.index_by_id["B10"] == 10
    rgb = resolve_subset(order, BandSpec("rgb"))
    full = resolve_subset(order, BandSpec("s2_12"))
    assert rgb.ids == ("B2", "B3", "B4")
    assert rgb.indices == (1, 2, 3)
    assert 10 not in full.indices
    assert full.ids[-2:] == ("B11", "B12")


def test_missing_description_raises() -> None:
    """An empty description is refused."""
    broken = list(_HYPOTHESIS)
    broken[11] = "   "
    with pytest.raises(ValueError, match="empty description"):
        verify_band_order(broken)


def test_missing_b10_raises() -> None:
    """A stack with no B10 token is refused."""
    with pytest.raises(ValueError, match="B10"):
        verify_band_order(_HYPOTHESIS[:-1])


def test_duplicate_id_raises() -> None:
    """Two bands claiming one id are refused."""
    duplicated = list(_HYPOTHESIS)
    duplicated[-1] = "B12 again"
    with pytest.raises(ValueError, match="duplicate"):
        verify_band_order(duplicated)


def test_leave_one_out_preserves_order() -> None:
    """Dropping B8A yields 11 ids and omits only B8A."""
    order = verify_band_order(_HYPOTHESIS)
    subset = resolve_subset(order, BandSpec("loo", dropped_band="B8A"))
    assert subset.ids == tuple(band for band in order.ids if band not in {"B8A", "B10"})
    assert "B8A" not in subset.ids
    assert len(subset.ids) == 11


def test_empty_descriptions_use_the_zenodo_order() -> None:
    """Thirteen blank descriptions become the corpus order with B10 last."""
    order = verify_band_order(coerce_descriptions([""] * 13))
    assert order.ids[-1] == "B10"
    assert resolve_subset(order, BandSpec("s2_12")).ids[-1] == "B12"


def test_s2_13_is_not_a_subset() -> None:
    """The full GeoTIFF, including B10, is not a trainable subset."""
    order = verify_band_order(_STANDARD)
    with pytest.raises(ValueError, match="unknown band set"):
        resolve_subset(order, BandSpec("s2_13"))
