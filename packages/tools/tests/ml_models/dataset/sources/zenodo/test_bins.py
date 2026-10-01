"""Tests for the Zenodo GSD bins."""

import pytest
from tools.ml_models.dataset.raw import BinSpec
from tools.ml_models.dataset.sources.zenodo.bins import (
    DEFAULT_BINS,
    EXTENT_M,
    NATIVE_SIDE,
    actual_gsd,
    bin_hw,
)


def test_default_bins_cover_native_and_flight_elevations() -> None:
    """Six bins: native 10 m plus boresight footprints at five elevations."""
    assert [item.bin_id for item in DEFAULT_BINS] == [
        "native10",
        "elevation5",
        "elevation15",
        "elevation25",
        "elevation35",
        "elevation45",
    ]
    native, *_rest, e45 = DEFAULT_BINS
    assert bin_hw(native) == (NATIVE_SIDE, NATIVE_SIDE)
    assert actual_gsd(native).lateral_m == pytest.approx(10.0)
    assert bin_hw(e45) == (34, 51)
    assert e45.elevation_deg == 45.0


def test_actual_gsd_is_extent_over_pixels() -> None:
    """Stored GSD reflects the rounded grid, not the nominal target."""
    for item in DEFAULT_BINS:
        height, width = bin_hw(item)
        pair = actual_gsd(item)
        assert pair.lateral_m == pytest.approx(EXTENT_M / width)
        assert pair.along_m == pytest.approx(EXTENT_M / height)


def test_bins_never_upsample_native() -> None:
    """A bin finer than 10 m is rejected."""
    with pytest.raises(ValueError, match="coarsen"):
        bin_hw(BinSpec("too-fine", 5.0, 5.0))
    with pytest.raises(ValueError):
        bin_hw(BinSpec("bad", float("nan"), 10.0))
