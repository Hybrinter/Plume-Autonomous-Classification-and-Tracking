"""Tests for the Zenodo GSD bins."""

import ast
from pathlib import Path

import pytest
from tools.ml_models.dataset.raw import BinSpec
from tools.ml_models.dataset.sources.zenodo import bins
from tools.ml_models.dataset.sources.zenodo.bins import (
    DEFAULT_BINS,
    EXTENT_M,
    NATIVE_SIDE,
    actual_gsd,
    bin_hw,
)


def test_default_bins_are_the_fixed_target_table() -> None:
    """Native 10 m plus five fixed GSD targets, each sizing a 1200 m grid."""
    assert [(item.bin_id, item.lateral_m, item.along_m) for item in DEFAULT_BINS] == [
        ("native10", 10.0, 10.0),
        ("gsd15", 15.0, 15.0),
        ("gsd20", 20.0, 20.0),
        ("gsd25", 25.0, 25.0),
        ("gsd30", 30.0, 30.0),
        ("gsd35", 35.0, 35.0),
    ]
    assert [bin_hw(item) for item in DEFAULT_BINS] == [
        (NATIVE_SIDE, NATIVE_SIDE),
        (80, 80),
        (60, 60),
        (48, 48),
        (40, 40),
        (34, 34),
    ]


def test_actual_gsd_is_extent_over_pixels() -> None:
    """Recorded GSD is the achieved extent over rounded output pixels."""
    for item in DEFAULT_BINS:
        height, width = bin_hw(item)
        pair = actual_gsd(item)
        assert pair.lateral_m == pytest.approx(EXTENT_M / width)
        assert pair.along_m == pytest.approx(EXTENT_M / height)
    gsd35 = DEFAULT_BINS[-1]
    assert bin_hw(gsd35) == (34, 34)
    assert actual_gsd(gsd35).lateral_m == pytest.approx(1200.0 / 34)
    assert actual_gsd(gsd35).along_m == pytest.approx(1200.0 / 34)


def test_bins_never_upsample_native() -> None:
    """Finer-than-10 m requests are rejected."""
    with pytest.raises(ValueError, match="coarsen"):
        bin_hw(BinSpec("too-fine", 9.0, 9.0))
    with pytest.raises(ValueError, match="coarsen"):
        bin_hw(BinSpec("bad", float("nan"), 10.0))
    with pytest.raises(ValueError, match="coarsen"):
        bin_hw(BinSpec("bad", float("inf"), 10.0))


def test_bins_have_no_flight_dependencies() -> None:
    """Bin construction stays free of flight camera, orbit, and frame code."""
    tree = ast.parse(Path(bins.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.split(".")[0] == "flight", alias.name
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] != "flight", node.module
    assert not hasattr(bins, "make_bins")
    assert not hasattr(bins, "FLIGHT_ELEVATIONS_DEG")
