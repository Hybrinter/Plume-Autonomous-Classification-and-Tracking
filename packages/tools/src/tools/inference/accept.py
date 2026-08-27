"""Model-artifact acceptance gate: manifest + hash + I/O contract + quality + latency.

Passing this gate admits a frozen .onnx artifact into data/models/. Training and
export live in tools.inference. The segmentor gate runs five checks:

  1. manifest: sidecar JSON with version / source SHA / dataset hash / I/O / SHA-256.
  2. hash: the artifact's SHA-256 equals the manifest digest.
  3. I/O contract: declared shapes match the flight inference contract.
  4. golden-scene IoU: predicted masks meet a minimum mean IoU.
  5. latency: worst per-frame inference time is within the budget.

The classifier gate swaps check 4 for image-level binary accuracy on golden
labels. A frame is positive when the logit is >= `logit_threshold` (default 0.0).

The artifact is RUN via an injected callable so the gate is testable without
onnxruntime. `onnx_inference_fn` and `onnx_classifier_inference_fn` build the
live callables.

Contains:
  - Manifest / GoldenScene / GoldenClassifierScene / AcceptanceReport.
  - ClassifierAcceptanceReport: classifier-specific quality fields.
  - load_manifest / accept_artifact / accept_classifier_artifact.
  - compute_iou: re-export from tools.inference.metrics.
  - onnx_inference_fn / onnx_classifier_inference_fn.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

# stdlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# third-party
import numpy as np
import torch

# internal
from flight.libs.types import Ok
from flight.payload.inference.verify import verify_io_contract, verify_model_hash

from tools.inference.metrics import compute_iou as compute_iou
from tools.inference.metrics import mean_binary_accuracy

Shape = tuple[int | None, ...]
InferenceFn = Callable[[torch.Tensor], torch.Tensor]
ClassifierInferenceFn = Callable[[torch.Tensor], float]

__all__ = [
    "AcceptanceReport",
    "ClassifierAcceptanceReport",
    "GoldenClassifierScene",
    "GoldenScene",
    "Manifest",
    "accept_artifact",
    "accept_classifier_artifact",
    "compute_iou",
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
    hash_ok = isinstance(verify_model_hash(artifact_path, manifest.sha256), Ok)
    contract_ok = isinstance(
        verify_io_contract(
            manifest.input_shape, manifest.output_shape, expected_input, expected_output
        ),
        Ok,
    )

    ious: list[float] = []
    worst_latency_ms = 0.0
    for scene in scenes:
        start = time.perf_counter()
        pred = run_inference(scene.input_tensor)
        worst_latency_ms = max(worst_latency_ms, (time.perf_counter() - start) * 1000.0)
        ious.append(compute_iou(pred, scene.gold_mask, iou_threshold))
    mean_iou = sum(ious) / float(len(ious)) if ious else 0.0
    iou_ok = bool(scenes) and mean_iou >= min_iou
    latency_ok = worst_latency_ms <= max_latency_ms

    accepted = hash_ok and contract_ok and iou_ok and latency_ok
    detail = (
        f"hash={hash_ok} contract={contract_ok} mean_iou={mean_iou:.3f}>={min_iou} "
        f"worst_latency_ms={worst_latency_ms:.1f}<={max_latency_ms}"
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
    try:
        import onnxruntime
    except ImportError as exc:  # pragma: no cover - exercised only where the SDK is absent
        raise ImportError("onnxruntime is required to run live model acceptance") from exc
    session = onnxruntime.InferenceSession(artifact_path)
    input_name = session.get_inputs()[0].name

    def _run(tensor: torch.Tensor) -> torch.Tensor:
        array = np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype=np.float32)
        logits = session.run(None, {input_name: array[np.newaxis, ...]})[0]
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
    hash_ok = isinstance(verify_model_hash(artifact_path, manifest.sha256), Ok)
    contract_ok = isinstance(
        verify_io_contract(
            manifest.input_shape, manifest.output_shape, expected_input, expected_output
        ),
        Ok,
    )

    preds: list[bool] = []
    labels: list[bool] = []
    worst_latency_ms = 0.0
    for scene in scenes:
        start = time.perf_counter()
        logit = float(run_inference(scene.input_tensor))
        worst_latency_ms = max(worst_latency_ms, (time.perf_counter() - start) * 1000.0)
        preds.append(logit >= logit_threshold)
        labels.append(scene.label_positive)
    accuracy = mean_binary_accuracy(tuple(preds), tuple(labels))
    accuracy_ok = bool(scenes) and accuracy >= min_accuracy
    latency_ok = worst_latency_ms <= max_latency_ms

    accepted = hash_ok and contract_ok and accuracy_ok and latency_ok
    detail = (
        f"hash={hash_ok} contract={contract_ok} accuracy={accuracy:.3f}>={min_accuracy} "
        f"worst_latency_ms={worst_latency_ms:.1f}<={max_latency_ms}"
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
    try:
        import onnxruntime
    except ImportError as exc:  # pragma: no cover - exercised only where the SDK is absent
        raise ImportError("onnxruntime is required to run live model acceptance") from exc
    session = onnxruntime.InferenceSession(artifact_path)
    input_name = session.get_inputs()[0].name

    def _run(tensor: torch.Tensor) -> float:
        array = np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype=np.float32)
        logits = session.run(None, {input_name: array[np.newaxis, ...]})[0]
        return float(np.asarray(logits, dtype=np.float32).reshape(-1)[0])

    return _run
