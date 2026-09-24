"""Dataset channel count and mask alignment."""

from __future__ import annotations

import numpy as np
from tools.original_dataset_analysis.bands import BandSpec, resolve_subset, verify_band_order
from tools.original_dataset_analysis.dataset import StudyDataset
from tools.original_dataset_analysis.index import TileRef
from tools.original_dataset_analysis.normalize import fit_band_stats

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
    image, label, mask = dataset[0]
    assert image.shape == (3, 60, 60)
    assert label.shape == (1,)
    assert float(label) == 1.0
    assert mask.shape == (1, 60, 60)
    assert float(mask.sum()) > 0.0
