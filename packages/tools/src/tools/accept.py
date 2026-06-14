"""Model-artifact acceptance gate (tools/): manifest + hash + I/O contract + IoU + latency.

Passing this gate is what admits a frozen .onnx artifact into data/models/ (spec Section 4).
Training lives in a separate model repo; this gate is the intake check in THIS repo. It runs the
five checks the spec requires:

  1. manifest: a sidecar JSON with version / model-repo SHA / dataset hash / I/O contract / SHA-256.
  2. hash: the artifact's SHA-256 equals the manifest digest (reuses flight.payload.model.verify).
  3. I/O contract: the manifest's declared input/output shapes match the flight inference contract.
  4. golden-scene IoU: the artifact's predicted masks meet a minimum mean IoU over a golden set.
  5. latency: the worst per-frame inference time is within the budget.

The artifact is RUN via an injected `run_inference` callable so the gate logic is fully testable
without onnxruntime (which is not installed here); `onnx_inference_fn` builds the real
onnxruntime-backed callable for live acceptance on a machine that has the SDK. IoU is pure numpy.

Contains:
  - Manifest / GoldenScene / AcceptanceReport: the data types.
  - load_manifest / compute_iou / accept_artifact: parsing, scoring, and the gate itself.
  - onnx_inference_fn: lazily build an onnxruntime-backed inference callable (live use only).

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

# internal
from flight.libs.types import Ok
from flight.payload.model.verify import verify_io_contract, verify_model_hash

Shape = tuple[int | None, ...]
InferenceFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True, slots=True)
class Manifest:
    """The sidecar manifest accompanying a frozen .onnx artifact."""

    version: str
    model_repo_sha: str
    dataset_hash: str
    input_shape: Shape
    output_shape: Shape
    sha256: str


@dataclass(frozen=True, slots=True)
class GoldenScene:
    """One golden evaluation case: a preprocessed input tensor and its expected mask."""

    input_tensor: np.ndarray  # np.ndarray[float32, (C, H, W)]
    gold_mask: np.ndarray  # np.ndarray[float32, (H, W)] in [0, 1]


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
    )


def compute_iou(pred_mask: np.ndarray, gold_mask: np.ndarray, threshold: float = 0.5) -> float:
    """Compute the binary intersection-over-union of a predicted vs golden mask (pure).

    Args:
        pred_mask: Predicted probability mask (H, W).
        gold_mask: Golden probability mask (H, W).
        threshold: Probability threshold for binarization.

    Returns:
        IoU in [0, 1]. Two empty masks (no positives in either) score 1.0 (perfect agreement).
    """
    pred = np.asarray(pred_mask) >= threshold
    gold = np.asarray(gold_mask) >= threshold
    intersection = float(np.logical_and(pred, gold).sum())
    union = float(np.logical_or(pred, gold).sum())
    if union == 0.0:
        return 1.0
    return intersection / union


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
    mean_iou = float(np.mean(ious)) if ious else 0.0
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

    def _run(tensor: np.ndarray) -> np.ndarray:
        logits = session.run(None, {input_name: tensor[np.newaxis, ...]})[0]
        probs = 1.0 / (1.0 + np.exp(-logits))
        return np.asarray(probs[0, 0], dtype=np.float32)

    return _run
