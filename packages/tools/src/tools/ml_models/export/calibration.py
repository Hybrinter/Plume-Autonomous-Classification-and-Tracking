"""Deterministic two-input calibration from finished training shards."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tools.ml_models.dataset.loader import ShardDataset
from tools.ml_models.dataset.manifest import load_manifest
from tools.ml_models.export.manifest import ModelManifest


def calibration_batches(
    dataset: str | Path,
    model: ModelManifest,
    samples: int = 32,
) -> list[dict[str, np.ndarray]]:
    """Round-robin across all train shards, using every row at most once.

    This includes matching encoded GSD for each image. It never calibrates on
    held-out data or invents geometry for synthetic noise.
    """
    if samples < 1:
        raise ValueError("calibration requires a positive count and a finished dataset")
    if not isinstance(dataset, str | Path) or not str(dataset).strip():
        raise ValueError("calibration requires exactly one finished dataset")
    manifest = load_manifest(Path(dataset) / "dataset.json")
    if (
        manifest.band_names != model.band_names
        or manifest.norm != model.norm
        or manifest.gsd_reference_m != model.gsd_reference_m
    ):
        raise ValueError("calibration preprocessing differs from the model")
    fixed_hw = (model.input_shape[2], model.input_shape[3])
    shards: list[ShardDataset] = []
    for count in manifest.shards:
        if count.task != model.kind or count.split != "train":
            continue
        if (fixed_hw[0] is not None and count.height != fixed_hw[0]) or (
            fixed_hw[1] is not None and count.width != fixed_hw[1]
        ):
            raise ValueError("calibration shard shape differs from the model")
        shards.append(
            ShardDataset(
                Path(dataset) / model.kind / "train" / f"{count.height}x{count.width}",
                model.gsd_reference_m,
                model.kind,
                channels=len(manifest.band_names),
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
