"""Tests for label-studio polygon annotation parsing and rasterization."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from tools.inference.annotations import (
    LABEL_SUFFIX,
    SMOKE_LABEL,
    annotation_stem,
    load_annotation_mask,
    load_polygons,
    parse_polygons,
    rasterize_polygons,
)

_SQUARE_POINTS = [[25.0, 25.0], [75.0, 25.0], [75.0, 75.0], [25.0, 75.0]]


def _annotation_dict(
    points: list[list[float]] | None = None,
    *,
    label: str = SMOKE_LABEL,
    result_type: str = "polygonlabels",
    completions: list[object] | None = None,
) -> dict[str, object]:
    """Build a minimal label-studio export dict."""
    if completions is not None:
        return {"completions": completions, "data": {}, "id": 1}
    return {
        "completions": [
            {
                "result": [
                    {
                        "type": result_type,
                        "value": {
                            "points": points or _SQUARE_POINTS,
                            "polygonlabels": [label],
                        },
                        "original_width": 120,
                        "original_height": 120,
                    }
                ]
            }
        ],
        "data": {},
        "id": 1,
    }


def test_annotation_stem_strips_suffix_and_json() -> None:
    """annotation_stem removes _features and .json but leaves a clean stem alone."""
    colon_name = "10003_2019-01-21T10:56:41.330Z_0_features.json"
    assert annotation_stem(colon_name) == "10003_2019-01-21T10:56:41.330Z_0"
    assert annotation_stem("alpha_features") == "alpha"
    assert annotation_stem("alpha") == "alpha"
    assert annotation_stem("alpha.json") == "alpha"
    assert LABEL_SUFFIX == "_features"


def test_parse_polygons_returns_smoke_polygons() -> None:
    """parse_polygons keeps percentage-space smoke polygons from a faithful export."""
    data = _annotation_dict(_SQUARE_POINTS)
    polygons = parse_polygons(data)
    assert len(polygons) == 1
    assert polygons[0].dtype == np.float32
    assert polygons[0].shape == (4, 2)
    assert float(polygons[0][0, 0]) == pytest.approx(25.0)


def test_parse_polygons_empty_and_invalid_inputs() -> None:
    """parse_polygons returns () for empty or non-dict inputs and skips bad results."""
    assert parse_polygons(_annotation_dict(completions=[])) == ()
    assert parse_polygons("not-a-dict") == ()
    assert parse_polygons(_annotation_dict(_SQUARE_POINTS, result_type="choices")) == ()
    assert parse_polygons(_annotation_dict(_SQUARE_POINTS, label="cloud")) == ()
    degenerate = _annotation_dict([[0.0, 0.0], [1.0, 1.0]])
    assert parse_polygons(degenerate) == ()


def test_rasterize_square_polygon_covers_quarter_area() -> None:
    """Percentage coordinates scale to roughly 25% coverage at any output size."""
    polygons = parse_polygons(_annotation_dict(_SQUARE_POINTS))
    for height, width in ((40, 40), (120, 120), (256, 256)):
        mask = rasterize_polygons(polygons, height, width)
        assert mask.shape == (height, width)
        assert mask.dtype == np.float32
        assert set(np.unique(mask).tolist()) <= {0.0, 1.0}
        assert float(mask.mean()) == pytest.approx(0.25, abs=0.03)


def test_rasterize_polygons_empty_returns_zeros() -> None:
    """No polygons yields an all-zero float32 mask of the requested shape."""
    mask = rasterize_polygons((), height=16, width=32)
    assert mask.shape == (16, 32)
    assert mask.dtype == np.float32
    assert float(mask.max()) == 0.0


def test_rasterize_polygons_unions_overlaps() -> None:
    """Overlapping polygons union without double-counting above 1."""
    first = np.array([[10.0, 10.0], [60.0, 10.0], [60.0, 60.0], [10.0, 60.0]], dtype=np.float32)
    second = np.array([[40.0, 40.0], [90.0, 40.0], [90.0, 90.0], [40.0, 90.0]], dtype=np.float32)
    mask = rasterize_polygons((first, second), height=64, width=64)
    assert float(mask.max()) == pytest.approx(1.0)
    assert set(np.unique(mask).tolist()) <= {0.0, 1.0}


def test_rasterize_polygons_rejects_non_positive_size() -> None:
    """Non-positive height or width raises ValueError."""
    polygons = parse_polygons(_annotation_dict(_SQUARE_POINTS))
    with pytest.raises(ValueError, match="mask size must be positive"):
        rasterize_polygons(polygons, height=0, width=8)
    with pytest.raises(ValueError, match="mask size must be positive"):
        rasterize_polygons(polygons, height=8, width=0)


def test_load_polygons_reads_json_file(tmp_path: Path) -> None:
    """load_polygons reads a Windows-legal annotation file from disk."""
    path = tmp_path / "sample_features.json"
    path.write_text(json.dumps(_annotation_dict(_SQUARE_POINTS)), encoding="utf-8")
    polygons = load_polygons(path)
    assert len(polygons) == 1
    assert polygons[0].shape == (4, 2)


def test_load_annotation_mask_shape_and_values(tmp_path: Path) -> None:
    """load_annotation_mask returns (1, H, W) float32 values in {0, 1}."""
    path = tmp_path / "sample_features.json"
    path.write_text(json.dumps(_annotation_dict(_SQUARE_POINTS)), encoding="utf-8")
    mask = load_annotation_mask(path, height=32, width=32)
    assert mask.shape == (1, 32, 32)
    assert mask.dtype == np.float32
    assert set(np.unique(mask).tolist()) <= {0.0, 1.0}
    assert float(mask.max()) == 1.0
