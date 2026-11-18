"""Tests for the late-fusion run checks."""

from __future__ import annotations

from flight.libs.types import Err, Ok
from tools.ml_models.fusion.study import check_gsd_reference


def test_gsd_reference_must_match_for_both_branches() -> None:
    """Equal references pass; a VIS or SWIR mismatch names the branch."""
    assert isinstance(check_gsd_reference(15.87, 15.87, 15.87), Ok)
    vis = check_gsd_reference(20.0, 15.87, 15.87)
    assert isinstance(vis, Err)
    assert "VIS" in vis.error
    swir = check_gsd_reference(15.87, 10.0, 15.87)
    assert isinstance(swir, Err)
    assert "SWIR" in swir.error
