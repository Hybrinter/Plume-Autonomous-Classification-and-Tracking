"""Model-artifact acceptance gate: manifest + hash + I/O contract + quality + latency.

Passing this gate admits a frozen .onnx artifact into data/models/. Training and
export live in tools.inference. The segmentor gate runs five checks:

  1. manifest: sidecar JSON with version / source SHA / dataset hash / I/O / SHA-256.
  2. hash: the artifact's SHA-256 equals the manifest digest.
  3. I/O contract: declared shapes match the flight inference contract.
  4. golden-scene IoU: predicted masks meet a minimum mean IoU.
  5. latency: worst per-frame inference time is within the budget. The report
     also carries the median and 95th percentile, which describe the run without
     the outlier sensitivity of a single worst sample.

The classifier gate swaps check 4 for image-level binary accuracy on golden
labels. A frame is positive when the logit is >= `logit_threshold` (default 0.0).

The artifact is RUN via an injected callable so the gate is testable without
onnxruntime. `onnx_inference_fn` and `onnx_classifier_inference_fn` build the
live callables.

Golden scenes come from a named split of a processed pack. Both quality checks
are vacuous without them: an empty scene list scores no IoU and no accuracy, so
the gate cannot pass. Reading them from the test split is what makes the gate
measure the artifact rather than only its manifest.

Contains:
  - Manifest / GoldenScene / GoldenClassifierScene / AcceptanceReport.
  - ClassifierAcceptanceReport: classifier-specific quality fields.
  - load_manifest / accept_artifact / accept_classifier_artifact / accept_kind.
  - load_golden_scenes / load_golden_classifier_scenes: scenes from a pack.
    Classifier scenes skip ``masks.npy`` and copy one image row at a time.
  - compute_iou: re-export from tools.inference.metrics.
  - onnx_inference_fn / onnx_classifier_inference_fn.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

# stdlib
import json
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# third-party
import numpy as np
import torch

# internal
from flight.libs.types import Ok
from flight.payload.inference.verify import verify_io_contract, verify_model_hash

from tools.inference.data import ProcessedPack, _row_image, load_processed_pack
from tools.inference.metrics import compute_iou as compute_iou
from tools.inference.metrics import mean_binary_accuracy
from tools.inference.ort_providers import resolve_ort_providers

Shape = tuple[int | None, ...]
InferenceFn = Callable[[torch.Tensor], torch.Tensor]
ClassifierInferenceFn = Callable[[torch.Tensor], float]
_LOG = logging.getLogger(__name__)

__all__ = [
    "AcceptanceReport",
    "ClassifierAcceptanceReport",
    "GoldenClassifierScene",
    "GoldenScene",
    "Manifest",
    "accept_artifact",
    "accept_classifier_artifact",
    "accept_kind",
    "compute_iou",
    "load_golden_classifier_scenes",
    "load_golden_scenes",
    "load_manifest",
    "onnx_classifier_inference_fn",
    "onnx_inference_fn",
]


@dataclass(frozen=True, slots=True)
class Manifest:
    """The sidecar manifest accompanying a frozen .onnx artifact."""

    version: str
    model_repo_sha: str
    dataset_hash: str
    input_shape: Shape
    output_shape: Shape
    sha256: str
    quantization: str = "fp32"


@dataclass(frozen=True, slots=True)
class GoldenScene:
    """One golden evaluation case: a preprocessed input tensor and its expected mask."""

    input_tensor: torch.Tensor  # torch.Tensor[float32, (C, H, W)]
    gold_mask: torch.Tensor  # torch.Tensor[float32, (H, W)] in [0, 1]


@dataclass(frozen=True, slots=True)
class GoldenClassifierScene:
    """One golden classifier case: a preprocessed tensor and a presence label."""

    input_tensor: torch.Tensor  # torch.Tensor[float32, (C, H, W)]
    label_positive: bool


@dataclass(frozen=True, slots=True)
class AcceptanceReport:
    """The acceptance outcome: per-check booleans + the aggregate accept decision."""

    hash_ok: bool
    contract_ok: bool
    mean_iou: float
    iou_ok: bool
    worst_latency_ms: float
    latency_ok: bool
    accepted: bool
    detail: str
    median_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class ClassifierAcceptanceReport:
    """Classifier intake outcome: hash, contract, accuracy, latency, accept flag."""

    hash_ok: bool
    contract_ok: bool
    accuracy: float
    accuracy_ok: bool
    worst_latency_ms: float
    latency_ok: bool
    accepted: bool
    detail: str
    median_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0


def load_manifest(path: str) -> Manifest:
    """Parse a manifest JSON sidecar into a Manifest.

    Args:
        path: Filesystem path to the manifest JSON.

    Returns:
        The parsed Manifest.

    Raises:
        OSError / json.JSONDecodeError / KeyError: on a missing/malformed manifest (tools-side
        engineering check; raising is appropriate).
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Manifest(
        version=str(data["version"]),
        model_repo_sha=str(data["model_repo_sha"]),
        dataset_hash=str(data["dataset_hash"]),
        input_shape=tuple(data["input_shape"]),
        output_shape=tuple(data["output_shape"]),
        sha256=str(data["sha256"]),
        quantization=str(data.get("quantization", "fp32")),
    )


def _latency_stats(samples: list[float]) -> tuple[float, float, float]:
    """Return the worst, median, and 95th-percentile latency of a sample list.

    Args:
        samples: Per-scene wall-clock times in milliseconds.

    Returns:
        tuple[float, float, float]: Worst, median, and p95 milliseconds, all zero
        for an empty list.

    Notes:
        The gate decides on the worst sample, which one descheduled scene can
        inflate. The median and p95 describe the same run without that
        sensitivity, so a budget can be set from typical cost rather than from
        the single unluckiest frame.
    """
    if not samples:
        return 0.0, 0.0, 0.0
    ordered = sorted(samples)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else 0.5 * (ordered[middle - 1] + ordered[middle])
    p95_index = min(len(ordered) - 1, int(math.ceil(0.95 * len(ordered))) - 1)
    return ordered[-1], median, ordered[max(p95_index, 0)]


def _split_indices(pack: ProcessedPack, split: str, limit: int) -> list[int]:
    """Return the sample indices of a named split, truncated to ``limit``."""
    table = {"train": pack.splits.train, "val": pack.splits.val, "test": pack.splits.test}
    if split not in table:
        raise ValueError(f"unknown split {split!r}; expected train, val, or test")
    indices = [int(index) for index in table[split]]
    return indices[:limit] if limit > 0 else indices


def load_golden_scenes(pack_dir: str, split: str = "test", limit: int = 0) -> list[GoldenScene]:
    """Read segmentor golden scenes from a named split of a processed pack.

    Args:
        pack_dir: Processed pack directory.
        split: ``train``, ``val``, or ``test`` (Default: "test").
        limit: Maximum scene count; ``0`` takes the whole split.

    Returns:
        list[GoldenScene]: One scene per sample, holding the ``(C, H, W)`` input
        and its ``(H, W)`` reference mask.

    Raises:
        ValueError: If ``split`` is not a known split name.
    """
    pack = load_processed_pack(pack_dir)
    return [
        GoldenScene(
            input_tensor=_row_image(pack.images, index).clone(),
            gold_mask=pack.masks[index, 0].clone(),
        )
        for index in _split_indices(pack, split, limit)
    ]


def load_golden_classifier_scenes(
    pack_dir: str, split: str = "test", limit: int = 0
) -> list[GoldenClassifierScene]:
    """Read classifier golden scenes from a named split of a processed pack.

    Args:
        pack_dir: Processed pack directory.
        split: ``train``, ``val``, or ``test`` (Default: "test").
        limit: Maximum scene count; ``0`` takes the whole split.

    Returns:
        list[GoldenClassifierScene]: One scene per sample, holding the
        ``(C, H, W)`` input and its plume-presence label.

    Raises:
        ValueError: If ``split`` is not a known split name.
    """
    pack = load_processed_pack(pack_dir, load_masks=False)
    return [
        GoldenClassifierScene(
            input_tensor=_row_image(pack.images, index).clone(),
            label_positive=bool(float(pack.labels[index, 0]) >= 0.5),
        )
        for index in _split_indices(pack, split, limit)
    ]


def _hash_and_contract(
    artifact_path: str,
    manifest: Manifest,
    expected_input: Shape,
    expected_output: Shape,
) -> tuple[bool, bool]:
    """Return ``(hash_ok, contract_ok)`` for one artifact and manifest."""
    hash_ok = isinstance(verify_model_hash(artifact_path, manifest.sha256), Ok)
    contract_ok = isinstance(
        verify_io_contract(
            manifest.input_shape, manifest.output_shape, expected_input, expected_output
        ),
        Ok,
    )
    return hash_ok, contract_ok


def _onnx_session(artifact_path: str) -> tuple[Any, str]:
    """Return an onnxruntime session and its first input name."""
    try:
        import onnxruntime
    except ImportError as exc:  # pragma: no cover - exercised only where the SDK is absent
        raise ImportError("onnxruntime is required to run live model acceptance") from exc
    providers = resolve_ort_providers()
    session = onnxruntime.InferenceSession(artifact_path, providers=providers)
    _LOG.info("onnxruntime providers: %s", session.get_providers())
    return session, session.get_inputs()[0].name


def _onnx_input(tensor: torch.Tensor) -> np.ndarray:
    """Return a batch-1 float32 array for an ONNX session."""
    array = np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype=np.float32)
    return array[np.newaxis, ...]


def accept_artifact(
    artifact_path: str,
    manifest: Manifest,
    scenes: list[GoldenScene],
    run_inference: InferenceFn,
    expected_input: Shape,
    expected_output: Shape,
    min_iou: float,
    max_latency_ms: float,
    iou_threshold: float = 0.5,
) -> AcceptanceReport:
    """Run the full acceptance gate and return a pass/fail report.

    Args:
        artifact_path: Path to the frozen .onnx artifact (hashed for the manifest check).
        manifest: The artifact's parsed manifest.
        scenes: The golden evaluation scenes (input tensor + expected mask).
        run_inference: Callable mapping an input tensor (C, H, W) to a predicted mask (H, W).
        expected_input: The required model input shape (the flight inference contract).
        expected_output: The required model output shape.
        min_iou: Minimum acceptable mean IoU over the golden scenes.
        max_latency_ms: Maximum acceptable worst-case per-scene inference time.
        iou_threshold: Probability threshold for IoU binarization.

    Returns:
        An AcceptanceReport; accepted is True iff all of hash / contract / IoU / latency pass.
    """
    hash_ok, contract_ok = _hash_and_contract(
        artifact_path, manifest, expected_input, expected_output
    )

    ious: list[float] = []
    latencies: list[float] = []
    for scene in scenes:
        start = time.perf_counter()
        pred = run_inference(scene.input_tensor)
        latencies.append((time.perf_counter() - start) * 1000.0)
        ious.append(compute_iou(pred, scene.gold_mask, iou_threshold))
    mean_iou = sum(ious) / float(len(ious)) if ious else 0.0
    iou_ok = bool(scenes) and mean_iou >= min_iou
    worst_latency_ms, median_latency_ms, p95_latency_ms = _latency_stats(latencies)
    latency_ok = worst_latency_ms <= max_latency_ms

    accepted = hash_ok and contract_ok and iou_ok and latency_ok
    detail = (
        f"hash={hash_ok} contract={contract_ok} mean_iou={mean_iou:.3f}>={min_iou} "
        f"worst_latency_ms={worst_latency_ms:.1f}<={max_latency_ms} "
        f"median_latency_ms={median_latency_ms:.1f} p95_latency_ms={p95_latency_ms:.1f}"
    )
    return AcceptanceReport(
        hash_ok=hash_ok,
        contract_ok=contract_ok,
        mean_iou=mean_iou,
        iou_ok=iou_ok,
        worst_latency_ms=worst_latency_ms,
        latency_ok=latency_ok,
        accepted=accepted,
        detail=detail,
        median_latency_ms=median_latency_ms,
        p95_latency_ms=p95_latency_ms,
    )


def onnx_inference_fn(artifact_path: str) -> InferenceFn:
    """Build an onnxruntime-backed inference callable for live acceptance (not used in CI).

    Args:
        artifact_path: Path to the frozen .onnx artifact.

    Returns:
        A callable mapping an input tensor (C, H, W) to a sigmoid mask (H, W).

    Raises:
        ImportError: If onnxruntime is not installed (acceptance must run where the SDK exists).
    """
    session, input_name = _onnx_session(artifact_path)

    def _run(tensor: torch.Tensor) -> torch.Tensor:
        logits = session.run(None, {input_name: _onnx_input(tensor)})[0]
        probs = 1.0 / (1.0 + np.exp(-logits))
        return torch.from_numpy(np.asarray(probs[0, 0], dtype=np.float32))

    return _run


def accept_classifier_artifact(
    artifact_path: str,
    manifest: Manifest,
    scenes: list[GoldenClassifierScene],
    run_inference: ClassifierInferenceFn,
    expected_input: Shape,
    expected_output: Shape,
    min_accuracy: float,
    max_latency_ms: float,
    logit_threshold: float = 0.0,
) -> ClassifierAcceptanceReport:
    """Run the classifier acceptance gate and return a pass/fail report.

    Args:
        artifact_path: Path to the frozen .onnx artifact (hashed for the manifest check).
        manifest: The artifact's parsed manifest.
        scenes: Golden tensors with presence labels.
        run_inference: Callable mapping (C, H, W) to a scalar logit.
        expected_input: Required input shape (flight contract).
        expected_output: Required output shape, typically (1, 1).
        min_accuracy: Minimum mean binary accuracy over the golden scenes.
        max_latency_ms: Maximum acceptable worst-case per-scene inference time.
        logit_threshold: Positive when logit >= this value. Default 0.0.

    Returns:
        ClassifierAcceptanceReport; accepted is True iff hash, contract, accuracy,
        and latency all pass.
    """
    hash_ok, contract_ok = _hash_and_contract(
        artifact_path, manifest, expected_input, expected_output
    )

    preds: list[bool] = []
    labels: list[bool] = []
    latencies: list[float] = []
    for scene in scenes:
        start = time.perf_counter()
        logit = float(run_inference(scene.input_tensor))
        latencies.append((time.perf_counter() - start) * 1000.0)
        preds.append(logit >= logit_threshold)
        labels.append(scene.label_positive)
    accuracy = mean_binary_accuracy(tuple(preds), tuple(labels))
    accuracy_ok = bool(scenes) and accuracy >= min_accuracy
    worst_latency_ms, median_latency_ms, p95_latency_ms = _latency_stats(latencies)
    latency_ok = worst_latency_ms <= max_latency_ms

    accepted = hash_ok and contract_ok and accuracy_ok and latency_ok
    detail = (
        f"hash={hash_ok} contract={contract_ok} accuracy={accuracy:.3f}>={min_accuracy} "
        f"worst_latency_ms={worst_latency_ms:.1f}<={max_latency_ms} "
        f"median_latency_ms={median_latency_ms:.1f} p95_latency_ms={p95_latency_ms:.1f}"
    )
    return ClassifierAcceptanceReport(
        hash_ok=hash_ok,
        contract_ok=contract_ok,
        accuracy=accuracy,
        accuracy_ok=accuracy_ok,
        worst_latency_ms=worst_latency_ms,
        latency_ok=latency_ok,
        accepted=accepted,
        detail=detail,
        median_latency_ms=median_latency_ms,
        p95_latency_ms=p95_latency_ms,
    )


def onnx_classifier_inference_fn(artifact_path: str) -> ClassifierInferenceFn:
    """Build an onnxruntime-backed classifier callable (logits, no sigmoid).

    Args:
        artifact_path: Path to the frozen classifier .onnx.

    Returns:
        A callable mapping (C, H, W) to a scalar logit.

    Raises:
        ImportError: If onnxruntime is not installed.
    """
    session, input_name = _onnx_session(artifact_path)

    def _run(tensor: torch.Tensor) -> float:
        logits = session.run(None, {input_name: _onnx_input(tensor)})[0]
        return float(np.asarray(logits, dtype=np.float32).reshape(-1)[0])

    return _run


def accept_kind(
    kind: str,
    artifact_path: str,
    manifest: Manifest,
    *,
    scenes_dir: str,
    scenes_split: str = "test",
    scenes_limit: int = 0,
    expected_input: Shape,
    height: int,
    width: int,
    min_iou: float,
    min_accuracy: float,
    max_latency_ms: float,
) -> AcceptanceReport | ClassifierAcceptanceReport:
    """Run the classifier or segmentor gate for ``kind``.

    Args:
        kind: ``classifier`` or ``segmentor``.
        artifact_path: Path to the frozen ``.onnx`` artifact.
        manifest: Parsed sidecar manifest.
        scenes_dir: Processed pack directory. Empty skips golden scenes.
        scenes_split: Split name when ``scenes_dir`` is set.
        scenes_limit: Maximum scenes; ``0`` takes the whole split.
        expected_input: Required input shape.
        height: Segmentor output height.
        width: Segmentor output width.
        min_iou: Segmentor IoU floor.
        min_accuracy: Classifier accuracy floor.
        max_latency_ms: Worst-case per-scene latency budget.

    Returns:
        AcceptanceReport | ClassifierAcceptanceReport: Kind-specific report.

    Raises:
        ValueError: If ``kind`` is unknown.
        ImportError: If the live ONNX callable cannot import onnxruntime.
    """
    match kind:
        case "segmentor":
            scenes = (
                load_golden_scenes(scenes_dir, scenes_split, scenes_limit) if scenes_dir else []
            )
            return accept_artifact(
                artifact_path,
                manifest,
                scenes,
                onnx_inference_fn(artifact_path),
                expected_input,
                (1, 1, height, width),
                min_iou,
                max_latency_ms,
            )
        case "classifier":
            clf_scenes = (
                load_golden_classifier_scenes(scenes_dir, scenes_split, scenes_limit)
                if scenes_dir
                else []
            )
            return accept_classifier_artifact(
                artifact_path,
                manifest,
                clf_scenes,
                onnx_classifier_inference_fn(artifact_path),
                expected_input,
                (1, 1),
                min_accuracy,
                max_latency_ms,
            )
        case _:
            raise ValueError(f"unknown train kind {kind!r}")
