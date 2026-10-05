"""Two-input ONNX quality acceptance boundary on a finished test split.

Artifact hash, band/domain/reference, and spatial-shape checks remain
fail-closed. Scoring itself is unavailable until the evidence evaluation
phase lands: after the configured validations pass, ``accept_artifact``
returns an explicit unavailable error instead of opening a session, running
inference, or writing an acceptance report.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
from flight.libs.types import Err, Result
from torch import nn

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
    dataset: str | Path,
    *,
    min_iou: float = 0.5,
    min_accuracy: float = 0.9,
    max_latency_ms: float = 20.0,
) -> Result[dict[str, object], str]:
    """Validate acceptance inputs, then refuse unavailable scoring.

    Thresholds retain the previous tools acceptance conventions. The scoring
    step is unavailable, so a valid invocation always ends in Err and no
    acceptance report is written.

    Args:
        artifact: ONNX artifact path.
        manifest: Model sidecar for the artifact.
        dataset: Exactly one finished dataset directory.
        min_iou: Minimum per-source mean IoU for segmentors.
        min_accuracy: Minimum per-source accuracy for classifiers.
        max_latency_ms: Worst allowed batch-one CPU latency.

    Returns:
        Result[dict[str, object], str]: Err for invalid inputs, and Err with
        the unavailable reason once every validation passes.
    """
    if (
        not all(math.isfinite(value) and 0 <= value <= 1 for value in (min_iou, min_accuracy))
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
    del artifact
    return Err(
        "acceptance scoring is unavailable until the evidence evaluation phase is implemented"
    )
