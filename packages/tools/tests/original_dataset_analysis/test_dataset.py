"""Dataset channel count and mask alignment."""

from __future__ import annotations

import numpy as np
import pytest
from tools.original_dataset_analysis.bands import BandSpec, resolve_subset, verify_band_order
from tools.original_dataset_analysis.dataset import StudyDataset
from tools.original_dataset_analysis.index import TileRef
from tools.original_dataset_analysis.normalize import BandStats, fit_band_stats

_ORDER = verify_band_order(
    (
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B8A",
        "B9",
        "B11",
        "B12",
        "B10",
    )
)


def test_rgb_item_has_three_channels_and_aligned_mask() -> None:
    """An RGB subset yields three image channels and a mask of the same side."""
    subset = resolve_subset(_ORDER, BandSpec("rgb"))
    native = np.arange(13 * 120 * 120, dtype=np.float32).reshape(13, 120, 120)
    polygon = np.array([[10.0, 10.0], [40.0, 10.0], [40.0, 40.0]], dtype=np.float32)
    tile = TileRef(
        stem="10_a",
        location_id="10",
        positive=True,
        member_name="positive/10_a.tif",
        polygons=(polygon,),
    )

    def reader(_tile: TileRef) -> np.ndarray:
        return native

    selected = native[list(subset.indices)]
    stats = fit_band_stats([selected])
    dataset = StudyDataset([tile], reader, subset, stats, 60)
    image, label, mask, annotated = dataset[0]
    assert image.shape == (3, 60, 60)
    assert label.shape == (1,)
    assert float(label) == 1.0
    assert mask.shape == (1, 60, 60)
    assert float(mask.sum()) > 0.0
    assert float(annotated) == 1.0


def test_missing_annotation_is_distinct_from_an_empty_mask() -> None:
    """A missing annotation file is not stored as a real empty mask."""
    subset = resolve_subset(_ORDER, BandSpec("rgb"))
    native = np.zeros((13, 120, 120), dtype=np.float32)
    selected = native[list(subset.indices)]
    stats = fit_band_stats([selected])

    def reader(_tile: TileRef) -> np.ndarray:
        return native

    def tile(stem: str, polygons: tuple[np.ndarray, ...] | None) -> TileRef:
        return TileRef(
            stem=stem,
            location_id="10",
            positive=False,
            member_name=f"negative/{stem}.tif",
            polygons=polygons,
        )

    missing = StudyDataset([tile("none", None)], reader, subset, stats, 60)
    empty = StudyDataset([tile("empty", ())], reader, subset, stats, 60)
    _, _, missing_mask, missing_flag = missing[0]
    _, _, empty_mask, empty_flag = empty[0]
    assert float(missing_mask.sum()) == 0.0
    assert float(empty_mask.sum()) == 0.0
    assert float(missing_flag) == 0.0
    assert float(empty_flag) == 1.0


def test_dataset_rejects_a_column_shaped_std() -> None:
    """Moments must be vectors, including the standard deviation."""
    subset = resolve_subset(_ORDER, BandSpec("rgb"))
    native = np.zeros((13, 120, 120), dtype=np.float32)
    stats = BandStats(
        mean=np.zeros(3, dtype=np.float32),
        std=np.ones((3, 1), dtype=np.float32),
    )

    def reader(_tile: TileRef) -> np.ndarray:
        return native

    tile = TileRef(
        stem="10_a",
        location_id="10",
        positive=True,
        member_name="positive/10_a.tif",
        polygons=(),
    )
    with pytest.raises(ValueError, match="must both be"):
        StudyDataset([tile], reader, subset, stats, 60)
