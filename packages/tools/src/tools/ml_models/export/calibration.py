"""Deterministic two-input calibration from finished training shards."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from tools.ml_models.dataset.loader import ShardDataset
from tools.ml_models.dataset.manifest import check_compatible, load_manifest
from tools.ml_models.export.manifest import ModelManifest


def calibration_batches(
    datasets: Sequence[str | Path],
    model: ModelManifest,
    samples: int = 32,
) -> list[dict[str, np.ndarray]]:
    """Round-robin across all train shards, using every row at most once.

    This includes matching encoded GSD for each image. It never calibrates on
    held-out data or invents geometry for synthetic noise.
    """
    if samples < 1 or not datasets:
        raise ValueError("calibration requires a positive count and finished datasets")
    manifests = [load_manifest(Path(dest) / "dataset.json") for dest in datasets]
    check_compatible(manifests)
    shards: list[ShardDataset] = []
    for dest, manifest in zip(datasets, manifests, strict=True):
        if (
            manifest.band_names != model.band_names
            or manifest.gsd_reference_m != model.gsd_reference_m
        ):
            raise ValueError("calibration preprocessing differs from the model")
        for count in manifest.shards:
            if count.task == model.kind and count.split == "train":
                shards.append(
                    ShardDataset(
                        Path(dest) / model.kind / "train" / f"{count.height}x{count.width}",
                        model.gsd_reference_m,
                        model.kind,
                    )
                )
    if not shards:
        raise ValueError("no training shards for calibration")
    batches: list[dict[str, np.ndarray]] = []
    row = 0
    while len(batches) < samples:
        available = False
        for shard in shards:
            if row >= len(shard):
                continue
            image, gsd, _target = shard[row]
            batches.append(
                {
                    "image": np.ascontiguousarray(image[None].numpy(), dtype=np.float32),
                    "gsd": np.ascontiguousarray(gsd[None].numpy(), dtype=np.float32),
                }
            )
            available = True
            if len(batches) == samples:
                break
        if not available:
            break
        row += 1
    return batches
