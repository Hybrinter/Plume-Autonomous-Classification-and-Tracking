"""Exhaustive evaluation of finished splits with mixed spatial extents.

Dataset and bin reports use every row once, without training resampling.
Combined metrics are an explicit dataset-weighted macro average.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from tools.ml_models.dataset.loader import ShardDataset
from tools.ml_models.dataset.manifest import DatasetManifest
from tools.ml_models.dataset.store import read_rows
from tools.ml_models.train.metrics import (
    classifier_metrics,
    compute_dice,
    compute_iou,
)


class _Scores:
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.logits: list[torch.Tensor] = []
        self.targets: list[torch.Tensor] = []
        self.overlap: list[tuple[float, float, float, float]] = []

    def add(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        logits, targets = logits.detach().cpu(), targets.detach().cpu()
        if logits.shape != targets.shape or not torch.isfinite(logits).all():
            raise ValueError("non-finite or misaligned evaluation output")
        if self.kind == "classifier":
            self.logits.append(logits.reshape(-1))
            self.targets.append(targets.reshape(-1))
        else:
            probabilities = torch.sigmoid(logits)
            for probability, logit, target in zip(probabilities, logits, targets, strict=True):
                self.overlap.append(
                    (
                        compute_iou(probability, target, 0.5),
                        compute_dice(probability, target, 0.5),
                        compute_iou(probability, target, 0.55),
                        float(torch.nn.functional.binary_cross_entropy_with_logits(logit, target)),
                    )
                )

    def result(self) -> dict[str, float]:
        if self.kind == "classifier":
            if not self.logits:
                raise ValueError("evaluation split contains no samples")
            result = asdict(
                classifier_metrics(
                    torch.cat(self.logits),
                    torch.cat(self.targets),
                )
            )
            return {key: float(value) for key, value in result.items()}
        if not self.overlap:
            raise ValueError("evaluation split contains no annotated samples")
        n = len(self.overlap)
        keys = ("mean_iou", "mean_dice", "mean_iou_blob_gate", "bce")
        return {"n": float(n)} | {
            key: sum(row[index] for row in self.overlap) / n for index, key in enumerate(keys)
        }


def evaluate(
    model: nn.Module,
    dests: Sequence[str | Path],
    manifests: Sequence[DatasetManifest],
    kind: str,
    split: str,
    batch_size: int,
    weights: Sequence[float],
    device: str,
) -> dict[str, object]:
    """Score full splits by source and bin, then weight source-level metrics.

    Input weights are positive and validated by TrainConfig. Training loads
    and verifies the manifests once before calling this function.
    """
    if len(dests) != len(manifests) or len(weights) != len(dests) or not dests:
        raise ValueError("evaluation dataset metadata is misaligned")
    previous_training = model.training
    model.eval()
    reports: list[dict[str, object]] = []
    summaries: list[dict[str, float]] = []
    try:
        with torch.no_grad():
            for dest, manifest in zip(dests, manifests, strict=True):
                scores = _Scores(kind)
                bins: dict[str, _Scores] = {}
                selected = [
                    shard
                    for shard in manifest.shards
                    if shard.task == kind and shard.split == split
                ]
                if not selected:
                    raise ValueError(f"no {kind}/{split} samples in {dest}")
                for shard in selected:
                    directory = Path(dest) / kind / split / f"{shard.height}x{shard.width}"
                    rows = read_rows(directory)
                    dataset = ShardDataset(
                        directory,
                        manifest.gsd_reference_m,
                        kind,
                        channels=len(manifest.band_names),
                    )
                    if len(rows) != len(dataset) or len(rows) != shard.n:
                        raise ValueError("evaluation shard count disagrees with manifest")
                    offset = 0
                    for images, gsd, targets in DataLoader(
                        dataset,
                        batch_size=batch_size,
                        shuffle=False,
                    ):
                        output = model(images.to(device), gsd.to(device))
                        scores.add(output, targets)
                        for index in range(len(images)):
                            row = rows[offset + index]
                            bin_id = row.bin_id or "unbinned"
                            bins.setdefault(bin_id, _Scores(kind)).add(
                                output[index : index + 1],
                                targets[index : index + 1],
                            )
                        offset += len(images)
                summary = scores.result()
                summaries.append(summary)
                reports.append(
                    {
                        "dataset": str(dest),
                        "source": manifest.source,
                        "dataset_hash": manifest.dataset_hash,
                        "metrics": summary,
                        "bins": {name: bucket.result() for name, bucket in sorted(bins.items())},
                    }
                )
    finally:
        model.train(previous_training)
    total = sum(weights)
    combined = {
        key: sum(weight * values[key] for weight, values in zip(weights, summaries, strict=True))
        / total
        for key in summaries[0]
        if key != "n"
    }
    combined["n"] = sum(values["n"] for values in summaries)
    return {
        "split": split,
        "aggregation": "dataset_weighted_macro",
        "datasets": reports,
        "combined": combined,
    }
