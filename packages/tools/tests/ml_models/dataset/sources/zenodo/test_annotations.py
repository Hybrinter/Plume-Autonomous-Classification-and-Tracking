"""Tests for Label Studio polygons and mask rasterization."""

import numpy as np
import pytest
from tools.ml_models.dataset.sources.zenodo.annotations import (
    parse_polygons,
    rasterize_percent_mask,
)


def _payload(points: list[list[float]]) -> dict[str, object]:
    return {
        "completions": [
            {
                "result": [
                    {
                        "type": "polygonlabels",
                        "value": {"polygonlabels": ["smoke"], "points": points},
                    }
                ]
            }
        ]
    }


def test_right_half_polygon_scales_with_width() -> None:
    """x in [50, 100] fills the right half of both a wide and a square mask."""
    polygons = parse_polygons(_payload([[50.0, 0.0], [100.0, 0.0], [100.0, 100.0], [50.0, 100.0]]))
    assert len(polygons) == 1
    square = rasterize_percent_mask(polygons, (120, 120))
    assert square.shape == (1, 120, 120)
    np.testing.assert_array_equal(square[0, :, :60], 0)
    np.testing.assert_array_equal(square[0, :, 60:], 1)
    wide = rasterize_percent_mask(polygons, (34, 51))
    np.testing.assert_array_equal(wide[0, :, :25], 0)
    np.testing.assert_array_equal(wide[0, :, 27:], 1)


def test_bottom_half_polygon_scales_with_height() -> None:
    """y in [50, 100] fills the bottom half of the mask."""
    polygons = parse_polygons(_payload([[0.0, 50.0], [100.0, 50.0], [100.0, 100.0], [0.0, 100.0]]))
    mask = rasterize_percent_mask(polygons, (120, 120))
    np.testing.assert_array_equal(mask[0, :60, :], 0)
    np.testing.assert_array_equal(mask[0, 60:, :], 1)


def test_no_smoke_label_is_an_empty_annotation() -> None:
    """A polygon without the smoke label yields no polygons."""
    value = {"polygonlabels": ["cloud"], "points": [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]]}
    payload = {"completions": [{"result": [{"type": "polygonlabels", "value": value}]}]}
    assert parse_polygons(payload) == ()


def test_malformed_annotations_are_rejected() -> None:
    """Bad payloads and bad vertices raise ValueError."""
    with pytest.raises(ValueError):
        parse_polygons([1, 2])
    with pytest.raises(ValueError):
        parse_polygons({"completions": {}})
    with pytest.raises(ValueError):
        parse_polygons(_payload([[0.0, 0.0], [200.0, 0.0], [10.0, 10.0]]))
    with pytest.raises(ValueError):
        parse_polygons(_payload([[0.0, 0.0], [10.0, 0.0]]))


def test_mask_rules_and_bounds() -> None:
    """The coverage rule is checked and masks stay binary."""
    polygons = parse_polygons(_payload([[50.0, 0.0], [100.0, 0.0], [100.0, 100.0], [50.0, 100.0]]))
    with pytest.raises(ValueError, match="rule"):
        rasterize_percent_mask(polygons, (10, 10), rule="bogus")
    mask = rasterize_percent_mask(polygons, (10, 10), rule="touch")
    assert set(np.unique(mask).tolist()) <= {0, 1}


def test_source_fitting_requires_both_dimensions() -> None:
    """source_hw and fitted_hw must be supplied together and positive."""
    polygons = parse_polygons(_payload([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0]]))
    with pytest.raises(ValueError, match="together"):
        rasterize_percent_mask(polygons, (120, 120), source_hw=(120, 121))
    with pytest.raises(ValueError, match="together"):
        rasterize_percent_mask(polygons, (120, 120), fitted_hw=(120, 120))
    with pytest.raises(ValueError, match="positive"):
        rasterize_percent_mask(polygons, (120, 120), source_hw=(120, 0), fitted_hw=(120, 120))


def test_source_fitting_maps_back_through_crop() -> None:
    """A 121-wide source cropped to 120 shifts a right-edge boundary left.

    A polygon covering x >= 60.75 source pixels leaves native column 60
    background and columns 61+ foreground.
    """
    boundary = 60.75 / 121 * 100
    polygons = parse_polygons(
        _payload([[boundary, 0.0], [100.0, 0.0], [100.0, 100.0], [boundary, 100.0]])
    )
    mask = rasterize_percent_mask(polygons, (120, 120), source_hw=(120, 121), fitted_hw=(120, 120))
    np.testing.assert_array_equal(mask[0, :, :61], 0)
    np.testing.assert_array_equal(mask[0, :, 61:], 1)


def test_source_fitting_maps_back_through_pad() -> None:
    """A 119-tall source padded to 120 replicates its last row in the mask."""
    polygons = parse_polygons(_payload([[0.0, 99.0], [100.0, 99.0], [100.0, 100.0], [0.0, 100.0]]))
    mask = rasterize_percent_mask(polygons, (120, 120), source_hw=(119, 120), fitted_hw=(120, 120))
    np.testing.assert_array_equal(mask[0, :118, :], 0)
    np.testing.assert_array_equal(mask[0, 118:, :], 1)
    np.testing.assert_array_equal(mask[0, 119, :], mask[0, 118, :])
    # A strip inside the last source row is not smeared into the padded row.
    edge = parse_polygons(_payload([[0.0, 99.6], [100.0, 99.6], [100.0, 100.0], [0.0, 100.0]]))
    narrow = rasterize_percent_mask(edge, (120, 120), source_hw=(119, 120), fitted_hw=(120, 120))
    np.testing.assert_array_equal(narrow[0, 119, :], 0)
