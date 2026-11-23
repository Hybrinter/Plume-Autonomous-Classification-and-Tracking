"""Measured training GSD coverage and single-dataset split validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tools.ml_models.dataset.manifest import DatasetManifest
from tools.ml_models.dataset.store import read_rows


def training_provenance(
    dataset: str | Path,
    manifest: DatasetManifest,
    task: str,
) -> dict[str, object]:
    """Return measured training coverage and reject groups reused across splits.

    Args:
        dataset: One finished dataset root.
        manifest: Verified dataset manifest.
        task: Classifier or segmentor task.

    Returns:
        dict[str, object]: Source identity, sample counts, shapes, and measured bounds.

    Raises:
        ValueError: On group leakage, malformed GSD, or missing measured training rows.
    """
    if not isinstance(dataset, str | Path) or not str(dataset).strip():
        raise ValueError("provenance requires exactly one finished dataset")
    groups: dict[str, str] = {}
    coverage_min = np.full(2, np.inf)
    coverage_max = np.full(2, -np.inf)
    shapes: set[tuple[int, int]] = set()
    counts = 0
    measured_count = 0
    bins: set[str] = set()
    for shard in manifest.shards:
        directory = Path(dataset) / shard.task / shard.split / f"{shard.height}x{shard.width}"
        rows = read_rows(directory)
        if len(rows) != shard.n:
            raise ValueError("row count disagrees with dataset manifest")
        for row in rows:
            old_split = groups.setdefault(row.group_id, shard.split)
            if old_split != shard.split:
                raise ValueError("related groups leak across dataset splits")
        if shard.task != task or shard.split != "train":
            continue
        gsd = np.load(directory / "gsd.npy", mmap_mode="r")
        if gsd.shape != (shard.n, 2) or not np.all(np.isfinite(gsd)) or np.any(gsd <= 0):
            raise ValueError("invalid training GSD array")
        measured = gsd[np.asarray([not row.gsd_nominal for row in rows], dtype=np.bool_)]
        if len(measured):
            coverage_min = np.minimum(coverage_min, measured.min(axis=0))
            coverage_max = np.maximum(coverage_max, measured.max(axis=0))
            measured_count += len(measured)
        shapes.add((shard.height, shard.width))
        counts += shard.n
        bins.update(row.bin_id for row in rows if row.bin_id)
    if counts < 1:
        raise ValueError("training task contains no samples")
    if measured_count < 1:
        raise ValueError("training task contains no measured GSD samples")
    return {
        "dataset": {
            "path": str(dataset),
            "source": manifest.source,
            "source_ref": manifest.source_ref,
            "dataset_hash": manifest.dataset_hash,
            "weight_table_id": manifest.weight_table_id,
            "bins": [item.bin_id for item in manifest.bins],
        },
        "train_samples": counts,
        "spatial_shapes": sorted(shapes),
        "gsd_bin_ids": sorted(bins),
        "gsd_min_m": coverage_min.tolist(),
        "gsd_max_m": coverage_max.tolist(),
        "gsd_reference_m": manifest.gsd_reference_m,
        "band_names": list(manifest.band_names),
        "in_channels": len(manifest.band_names),
        "norm": "unit",
    }
