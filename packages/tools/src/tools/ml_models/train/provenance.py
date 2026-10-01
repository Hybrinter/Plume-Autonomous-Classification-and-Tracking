"""Training-only GSD coverage and cross-dataset split validation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from tools.ml_models.dataset.manifest import DatasetManifest
from tools.ml_models.dataset.store import read_rows


def training_provenance(
    dests: Sequence[str | Path],
    manifests: Sequence[DatasetManifest],
    task: str,
) -> dict[str, object]:
    """Return persisted training geometry and reject reused groups across splits.

    Only sources with a nonempty source reference share a group namespace
    across dataset roots. Synthetic test sources do not claim a shared origin.
    """
    if len(dests) != len(manifests) or not dests:
        raise ValueError("misaligned dataset provenance")
    groups: dict[tuple[str, str, str], str] = {}
    coverage_min = np.full(2, np.inf)
    coverage_max = np.full(2, -np.inf)
    shapes: set[tuple[int, int]] = set()
    counts = 0
    bins: set[str] = set()
    sources: list[dict[str, object]] = []
    for dest, manifest in zip(dests, manifests, strict=True):
        sources.append(
            {
                "path": str(dest),
                "source": manifest.source,
                "source_ref": manifest.source_ref,
                "dataset_hash": manifest.dataset_hash,
                "weight_table_id": manifest.weight_table_id,
                "bins": [item.bin_id for item in manifest.bins],
            }
        )
        for shard in manifest.shards:
            if shard.task != task:
                continue
            directory = Path(dest) / task / shard.split / f"{shard.height}x{shard.width}"
            rows = read_rows(directory)
            if len(rows) != shard.n:
                raise ValueError("row count disagrees with dataset manifest")
            for row in rows:
                namespace = manifest.source_ref or str(Path(dest).resolve())
                key = (manifest.source, namespace, row.group_id)
                old_split = groups.setdefault(key, shard.split)
                if old_split != shard.split:
                    raise ValueError("related groups leak across combined dataset splits")
            if shard.split != "train":
                continue
            gsd = np.load(directory / "gsd.npy", mmap_mode="r")
            if gsd.shape != (shard.n, 2) or not np.all(np.isfinite(gsd)) or np.any(gsd <= 0):
                raise ValueError("invalid training GSD array")
            coverage_min = np.minimum(coverage_min, gsd.min(axis=0))
            coverage_max = np.maximum(coverage_max, gsd.max(axis=0))
            shapes.add((shard.height, shard.width))
            counts += shard.n
            bins.update(row.bin_id for row in rows if row.bin_id)
    if counts < 1:
        raise ValueError("training task contains no samples")
    return {
        "datasets": sources,
        "train_samples": counts,
        "spatial_shapes": sorted(shapes),
        "gsd_bin_ids": sorted(bins),
        "gsd_min_m": coverage_min.tolist(),
        "gsd_max_m": coverage_max.tolist(),
        "gsd_reference_m": manifests[0].gsd_reference_m,
        "band_names": list(manifests[0].band_names),
        "in_channels": len(manifests[0].band_names),
        "norm": "unit",
    }
