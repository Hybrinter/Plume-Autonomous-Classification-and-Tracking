"""Tests for the flight shape contract and GSD coverage helpers."""

import math

import pytest
from flight.libs.types import Err, Ok
from flight.payload.inference.contract import Shape, verify_conditioned_shapes
from tools.ml_models.export.contract import (
    GsdCoverage,
    coverage_ok,
    required_gsd_coverage,
)

_TILE = (193, 258)


def test_conditioned_classifier_shapes_pass() -> None:
    """Dynamic-batch image/gsd/logits shapes satisfy the classifier contract."""
    result = verify_conditioned_shapes(
        (None, 3, 193, 258), (None, 2), (None, 1), 3, _TILE, "classifier"
    )
    assert isinstance(result, Ok)
    dynamic = verify_conditioned_shapes(
        (None, 3, None, None), (None, 2), (None, 1), 3, _TILE, "classifier"
    )
    assert isinstance(dynamic, Ok)


def test_conditioned_segmentor_shapes_pass() -> None:
    """The segmentor accepts dynamic or tile-exact spatial output dims."""
    assert isinstance(
        verify_conditioned_shapes(
            (None, 3, 193, 258), (None, 2), (None, 1, 193, 258), 3, _TILE, "segmentor"
        ),
        Ok,
    )
    assert isinstance(
        verify_conditioned_shapes(
            (None, 3, None, None), (None, 2), (None, 1, None, None), 3, _TILE, "segmentor"
        ),
        Ok,
    )


@pytest.mark.parametrize(
    ("image", "gsd", "output", "kind"),
    [
        ((1, 3, 193, 258), (None, 2), (None, 1), "classifier"),
        ((None, 4, 193, 258), (None, 2), (None, 1), "classifier"),
        ((None, 3, 193), (None, 2), (None, 1), "classifier"),
        ((None, 3, 193, 258), (None, 3), (None, 1), "classifier"),
        ((None, 3, 193, 258), (1, 2), (None, 1), "classifier"),
        ((None, 3, 193, 258), (None, 2), (None, 2), "classifier"),
        ((None, 3, 193, 258), (None, 2), (1, 1), "classifier"),
        ((None, 3, 193, 258), (None, 2), (None, 1, 200, 258), "segmentor"),
        ((None, 3, 193, 258), (None, 2), (None, 2, 193, 258), "segmentor"),
        ((None, 3, 193, 258), (None, 2), (None, 1), "segmentor"),
        ((None, 3, 193, 258), (None, 2), (None, 1), "unknown"),
    ],
    ids=[
        "fixed-image-batch",
        "wrong-channels",
        "image-rank3",
        "gsd-width3",
        "fixed-gsd-batch",
        "classifier-two-classes",
        "fixed-output-batch",
        "segmentor-bad-height",
        "segmentor-two-channels",
        "segmentor-rank2-output",
        "unknown-kind",
    ],
)
def test_conditioned_shapes_reject_bad_io(
    image: Shape, gsd: Shape, output: Shape, kind: str
) -> None:
    """Malformed I/O shapes return Err(MODEL_CORRUPT)."""
    result = verify_conditioned_shapes(image, gsd, output, 3, _TILE, kind)
    assert isinstance(result, Err)


def test_required_coverage_matches_whole_grid_geometry() -> None:
    """The required coverage spans the configured science elevations."""
    required = required_gsd_coverage()
    assert required is not None
    assert required.altitude_m == 460_000.0
    assert all(v > 0 for v in (*required.minimum_m, *required.maximum_m))
    assert required.minimum_m[0] < required.maximum_m[0]
    assert required.minimum_m[1] < required.maximum_m[1]
    assert required.maximum_m == pytest.approx((23.976841, 37.979267), rel=1e-4)


def test_coverage_ok() -> None:
    """coverage_ok enforces a spanning interval with a 1 mm absolute tolerance."""
    required = GsdCoverage(minimum_m=(16.0, 16.0), maximum_m=(24.0, 38.0), altitude_m=460_000.0)
    assert coverage_ok((15.0, 15.0), (40.0, 40.0), required)
    assert coverage_ok((16.0, 16.0), (24.0, 38.0), required)
    assert coverage_ok((16.0 + 1e-3, 16.0), (24.0 - 1e-3, 38.0), required)
    assert not coverage_ok((17.0, 15.0), (40.0, 40.0), required)
    assert not coverage_ok((15.0, 15.0), (23.0, 40.0), required)
    assert not coverage_ok((15.0, 15.0), (40.0, 30.0), required)
    assert not coverage_ok((15.0, 15.0), (14.0, 40.0), required)
    assert not coverage_ok((0.0, 15.0), (40.0, 40.0), required)
    assert not coverage_ok((math.inf, 15.0), (40.0, 40.0), required)


def test_coverage_ok_absolute_tolerance() -> None:
    """A 1 cm shortfall on the 37.979267 m upper bound is rejected, not rounding."""
    required = GsdCoverage(
        minimum_m=(15.9006, 15.9337), maximum_m=(23.9768, 37.979267), altitude_m=460_000.0
    )
    assert not coverage_ok((15.0, 15.0), (30.0, 37.969267), required)
    assert coverage_ok((15.0, 15.0), (30.0, 37.979267 - 1e-3), required)
