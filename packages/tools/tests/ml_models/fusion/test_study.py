"""Tests for the late-fusion run checks."""

from __future__ import annotations

from flight.libs.types import Err, Ok
from tools.ml_models.fusion.late import DEFAULT_VIS_WEIGHTS
from tools.ml_models.fusion.study import check_bands, check_gsd_reference, check_weights


def test_gsd_reference_must_match_for_both_branches() -> None:
    """Equal references pass; a VIS or SWIR mismatch names the branch."""
    assert isinstance(check_gsd_reference(15.87, 15.87, 15.87), Ok)
    vis = check_gsd_reference(20.0, 15.87, 15.87)
    assert isinstance(vis, Err)
    assert "VIS" in vis.error
    swir = check_gsd_reference(15.87, 10.0, 15.87)
    assert isinstance(swir, Err)
    assert "SWIR" in swir.error


def test_bands_must_match_paired_dataset() -> None:
    """VIS then SWIR bands must equal the paired bands in order and count."""
    vis = ("red", "green", "blue")
    swir = ("swir1", "swir2")
    assert isinstance(check_bands(vis, swir, vis + swir), Ok)
    swapped = check_bands(("blue", "green", "red"), swir, vis + swir)
    assert isinstance(swapped, Err)
    assert "('blue', 'green', 'red')" in swapped.error
    other = check_bands(("red", "green", "nir"), swir, vis + swir)
    assert isinstance(other, Err)
    count = check_bands(vis, ("swir1",), vis + swir)
    assert isinstance(count, Err)


def test_weights_must_be_valid_grid() -> None:
    """The default grid passes; empty, out-of-range, NaN, and key clashes fail."""
    assert isinstance(check_weights(DEFAULT_VIS_WEIGHTS), Ok)
    for bad in ((), (1.5,), (-0.1,), (float("nan"),), (0.333, 0.334)):
        assert isinstance(check_weights(bad), Err)
