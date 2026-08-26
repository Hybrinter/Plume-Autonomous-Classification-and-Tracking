"""Builders for synthetic smoke-plume data.

Two entry points:
- ``build_synthetic_prepared_tree`` produces the post-prepare_dataset.py layout
  consumed by the training code.
- ``build_synthetic_zenodo_source`` produces the pre-prepare_dataset.py layout
  consumed by scripts/prepare_dataset.py.

Both write real 13-band uint16 GeoTIFFs at 120x120 (so Sentinel-2 band indexing
[2,3,4,8] works) and Label-Studio-style JSON polygons with a scale factor that
survives the 1.2x multiplier in SmokePlumeSegmentationDataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio as rio
from rasterio.transform import from_origin

TIF_COUNT = 13
TIF_SIZE = 120


def _write_tif(path: Path, rng: np.random.Generator) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = rng.integers(0, 3000, size=(TIF_COUNT, TIF_SIZE, TIF_SIZE), dtype=np.uint16)
    profile = {
        "driver": "GTiff",
        "width": TIF_SIZE,
        "height": TIF_SIZE,
        "count": TIF_COUNT,
        "dtype": "uint16",
        "transform": from_origin(0, 0, 10, 10),
        "crs": "EPSG:4326",
    }
    with rio.open(path, "w", **profile) as dst:
        dst.write(data)


def _write_label_json(path: Path, tif_basename: str) -> None:
    """Write a Label-Studio-style polygon label file.

    Polygon is a 50x50 square near the top-left so after the 1.2x scale the
    rasterized mask stays within the 120x120 frame.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "completions": [
            {
                "result": [
                    {
                        "value": {
                            "points": [
                                [10, 10],
                                [60, 10],
                                [60, 60],
                                [10, 60],
                            ]
                        }
                    }
                ]
            }
        ],
        "data": {"image": f"/data/upload/1-{tif_basename.replace('.tif', '.png')}"},
    }
    path.write_text(json.dumps(payload))


def build_synthetic_prepared_tree(
    root: Path,
    n_sites_per_split: int = 3,
    n_times_per_site: int = 2,
    seed: int = 42,
) -> None:
    """Produce the post-prepare_dataset.py layout under ``root``."""
    rng = np.random.default_rng(seed)
    root = Path(root)
    for split in ("train", "val", "test"):
        for site_idx in range(n_sites_per_split):
            for cls in ("positive", "negative"):
                for t in range(n_times_per_site):
                    basename = f"site{split}{site_idx}_t{t:03d}.tif"
                    _write_tif(root / "classification" / split / cls / basename, rng)
                    _write_tif(root / "segmentation" / split / "images" / cls / basename, rng)
                    if cls == "positive":
                        _write_label_json(
                            root / "segmentation" / split / "labels" / f"{basename}.json",
                            basename,
                        )


def build_synthetic_zenodo_source(
    root: Path,
    n_sites: int = 6,
    n_times_per_site: int = 2,
    seed: int = 42,
) -> None:
    """Produce the pre-prepare_dataset.py Zenodo-style source layout.

    Creates:
      root/images/{positive,negative}/*.tif
      root/segmentation_labels/*.json
    """
    rng = np.random.default_rng(seed)
    root = Path(root)
    for site_idx in range(n_sites):
        for cls in ("positive", "negative"):
            for t in range(n_times_per_site):
                basename = f"site{site_idx:03d}_t{t:03d}.tif"
                _write_tif(root / "images" / cls / basename, rng)
                if cls == "positive":
                    _write_label_json(
                        root / "segmentation_labels" / f"{basename}.json",
                        basename,
                    )
