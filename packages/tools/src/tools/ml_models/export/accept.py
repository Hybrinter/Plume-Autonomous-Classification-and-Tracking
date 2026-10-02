"""Exhaustive two-input ONNX quality acceptance on a finished test split."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from flight.libs.types import Err
from torch import nn

from tools.ml_models.dataset.manifest import check_compatible, load_manifest
from tools.ml_models.export.manifest import ModelManifest
from tools.ml_models.export.session import open_session
from tools.ml_models.train.evaluate import evaluate


class _OnnxModel(nn.Module):
    def __init__(self, artifact: Path, manifest: ModelManifest) -> None:
        super().__init__()
        opened = open_session(artifact, manifest)
        if isinstance(opened, Err):
            raise ValueError(opened.error)
        self.session = opened.value
        self.latencies_ms: list[float] = []

    def forward(self, image: torch.Tensor, gsd: torch.Tensor) -> torch.Tensor:
        start = time.perf_counter()
        outputs = self.session.run(
            None,
            {
                "image": image.detach().cpu().numpy().astype(np.float32),
                "gsd": gsd.detach().cpu().numpy().astype(np.float32),
            },
        )
        elapsed = (time.perf_counter() - start) * 1000
        if len(outputs) != 1:
            raise ValueError("model returned more than one logit output")
        self.latencies_ms.append(elapsed)
        return torch.from_numpy(np.asarray(outputs[0], dtype=np.float32))


def accept_artifact(
    artifact: str | Path,
    manifest: ModelManifest,
    datasets: Sequence[str | Path],
    *,
    min_iou: float = 0.5,
    min_accuracy: float = 0.9,
    max_latency_ms: float = 20.0,
) -> dict[str, object]:
    """Gate hash/contract, exhaustive task metrics and worst batch-one latency.

    Thresholds retain the previous tools acceptance conventions. CPU timings
    here are not a claim about the 64-tile on-board latency budget.
    """
    if (
        not all(math.isfinite(value) and 0 <= value <= 1 for value in (min_iou, min_accuracy))
        or not math.isfinite(max_latency_ms)
        or max_latency_ms <= 0
    ):
        raise ValueError("invalid acceptance thresholds")
    manifests = [load_manifest(Path(dest) / "dataset.json") for dest in datasets]
    check_compatible(manifests)
    if any(
        item.band_names != manifest.band_names or item.gsd_reference_m != manifest.gsd_reference_m
        for item in manifests
    ):
        raise ValueError("acceptance datasets disagree with model preprocessing")
    model = _OnnxModel(Path(artifact), manifest)
    report = evaluate(
        model,
        datasets,
        manifests,
        manifest.kind,
        "test",
        1,
        (1.0,) * len(datasets),
        "cpu",
    )
    reports = report["datasets"]
    assert isinstance(reports, list)
    metric = "accuracy" if manifest.kind == "classifier" else "mean_iou"
    threshold = min_accuracy if manifest.kind == "classifier" else min_iou
    quality_ok = True
    for source in reports:
        assert isinstance(source, dict)
        values = source["metrics"]
        assert isinstance(values, dict)
        quality_ok = quality_ok and float(values[metric]) >= threshold
    worst = max(model.latencies_ms, default=math.inf)
    latency_ok = math.isfinite(worst) and worst <= max_latency_ms
    return {
        "hash_ok": True,
        "contract_ok": True,
        "quality_ok": quality_ok,
        "latency_ok": latency_ok,
        "accepted": quality_ok and latency_ok,
        "metric": metric,
        "threshold": threshold,
        "worst_batch_one_latency_ms": worst,
        "max_latency_ms": max_latency_ms,
        "evaluation": report,
    }
