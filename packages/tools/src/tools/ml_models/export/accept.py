"""Two-input ONNX quality acceptance boundary on a finished test split.

Artifact integrity and preprocessing checks precede exhaustive shared test
evaluation. The legacy classifier accuracy and all-annotated-image IoU gates
remain explicit acceptance policies, not model-selection headline metrics.
Batch-one CPU timing is not target-hardware qualification.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from flight.libs.types import Err, Ok, Result
from torch import nn

from tools.ml_models.analysis.config import EvaluationConfig
from tools.ml_models.analysis.evaluate import evaluate_split
from tools.ml_models.dataset.manifest import load_manifest
from tools.ml_models.export.manifest import ModelManifest
from tools.ml_models.export.session import open_session


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
        try:
            outputs = self.session.run(
                None,
                {
                    "image": image.detach().cpu().numpy().astype(np.float32),
                    "gsd": gsd.detach().cpu().numpy().astype(np.float32),
                },
            )
        except Exception as exc:
            raise ValueError(f"ONNX inference failed: {exc}") from exc
        elapsed = (time.perf_counter() - start) * 1000
        if len(outputs) != 1:
            raise ValueError("model returned more than one logit output")
        self.latencies_ms.append(elapsed)
        output = np.asarray(outputs[0])
        if output.dtype != np.float32:
            raise ValueError("model logit output must be float32")
        return torch.from_numpy(output)


def accept_artifact(
    artifact: str | Path,
    manifest: ModelManifest,
    dataset: str | Path,
    *,
    min_iou: float = 0.5,
    min_accuracy: float = 0.9,
    max_latency_ms: float = 20.0,
) -> Result[dict[str, object], str]:
    """Gate integrity, exhaustive shared quality evidence, and CPU latency.

    Thresholds retain the previous tools acceptance conventions. Segmentor
    acceptance consumes the explicitly named all-annotated-image IoU adapter,
    not the positive-truth-image checkpoint-selection metric.

    Args:
        artifact: ONNX artifact path.
        manifest: Model sidecar for the artifact.
        dataset: Exactly one finished dataset directory.
        min_iou: Minimum all-annotated-image mean IoU for the dataset.
        min_accuracy: Minimum classifier accuracy for the dataset.
        max_latency_ms: Worst allowed batch-one CPU latency.

    Returns:
        Result[dict[str, object], str]: Measured acceptance or rejection;
        Err for invalid inputs, inference failure, or unavailable quality.
        This function does not write a report; the CLI owns publication.
    """
    if (
        not all(
            not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1
            for value in (min_iou, min_accuracy)
        )
        or isinstance(max_latency_ms, bool)
        or not math.isfinite(max_latency_ms)
        or max_latency_ms <= 0
    ):
        return Err("invalid acceptance thresholds")
    if not isinstance(dataset, str | Path) or not str(dataset).strip():
        return Err("acceptance requires exactly one finished dataset")
    try:
        dataset_manifest = load_manifest(Path(dataset) / "dataset.json")
    except (OSError, ValueError) as exc:
        return Err(str(exc))
    if (
        dataset_manifest.band_names != manifest.band_names
        or dataset_manifest.norm != manifest.norm
        or dataset_manifest.gsd_reference_m != manifest.gsd_reference_m
    ):
        return Err("acceptance dataset disagrees with model preprocessing")
    input_hw = manifest.input_shape[2:]
    for shard in dataset_manifest.shards:
        if shard.task != manifest.kind or shard.split != "test":
            continue
        if input_hw[0] is not None and shard.height != input_hw[0]:
            return Err("dataset shard height disagrees with model input_shape")
        if input_hw[1] is not None and shard.width != input_hw[1]:
            return Err("dataset shard width disagrees with model input_shape")
    if not any(
        shard.task == manifest.kind and shard.split == "test" for shard in dataset_manifest.shards
    ):
        return Err("acceptance dataset has no eligible test rows")
    try:
        model = _OnnxModel(Path(artifact), manifest)
    except (OSError, ValueError, RuntimeError) as exc:
        return Err(f"cannot validate acceptance artifact: {exc}")
    evaluated = evaluate_split(
        model,
        Path(dataset),
        dataset_manifest,
        EvaluationConfig(kind=manifest.kind, split="test", batch_size=1, device="cpu"),
    )
    if isinstance(evaluated, Err):
        return Err(f"acceptance evaluation unavailable: {evaluated.error}")
    evidence = evaluated.value
    name = (
        "accuracy" if manifest.kind == "classifier" else "foreground_iou_mean_all_annotated_images"
    )
    quality = next((metric for metric in evidence.metrics if metric.name == name), None)
    if (
        quality is None
        or quality.status != "AVAILABLE"
        or quality.value is None
        or quality.support.n < 1
    ):
        return Err(f"acceptance quality metric {name} is unavailable")
    if len(model.latencies_ms) != evidence.support.n or not all(
        math.isfinite(value) and value >= 0.0 for value in model.latencies_ms
    ):
        return Err("acceptance latency evidence is unavailable or misaligned")
    worst = max(model.latencies_ms)
    threshold = min_accuracy if manifest.kind == "classifier" else min_iou
    quality_ok = quality.value >= threshold
    latency_ok = worst <= max_latency_ms
    return Ok(
        {
            "sha256": manifest.sha256,
            "hash_ok": True,
            "contract_ok": True,
            "quality_ok": quality_ok,
            "latency_ok": latency_ok,
            "accepted": quality_ok and latency_ok,
            "metric": name,
            "quality_policy": (
                "legacy_classifier_accuracy"
                if manifest.kind == "classifier"
                else "legacy_all_annotated_image_iou"
            ),
            "threshold": threshold,
            "worst_batch_one_latency_ms": worst,
            "max_latency_ms": max_latency_ms,
            "evaluation": asdict(evidence),
        }
    )
