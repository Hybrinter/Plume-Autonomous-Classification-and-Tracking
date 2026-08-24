# tools.accept

**Source:** `packages/tools/src/tools/accept.py`
**Kind:** module

## Purpose

The acceptance gate checks a frozen ONNX artifact before it enters `data/models/`. It runs
manifest, hash, I/O contract, golden-scene IoU, and latency checks.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `Manifest` | class | Sidecar JSON fields for a frozen artifact |
| `GoldenScene` | class | Input tensor and expected mask for IoU scoring |
| `AcceptanceReport` | class | Per-check booleans and aggregate accept flag |
| `load_manifest` | function | Parse manifest JSON |
| `compute_iou` | function | Binary IoU between predicted and golden masks |
| `accept_artifact` | function | Run the full gate and return a report |
| `onnx_inference_fn` | function | Build an onnxruntime-backed inference callable |

## Inputs and outputs

**`load_manifest(path) -> Manifest`**

- Output: parsed `Manifest`.
- Raises on missing or malformed JSON.

**`compute_iou(pred_mask, gold_mask, threshold=0.5) -> float`**

- Output: IoU in [0, 1]. Two empty masks score 1.0.

**`accept_artifact(artifact_path, manifest, scenes, run_inference, ...) -> AcceptanceReport`**

- Inputs: artifact path, manifest, golden scenes, inference callable, expected shapes, IoU and
  latency thresholds.
- Output: `AcceptanceReport` with `accepted` true only when all checks pass.

**`onnx_inference_fn(artifact_path) -> InferenceFn`**

- Output: callable mapping `(C, H, W)` tensor to sigmoid mask `(H, W)`.
- Raises `ImportError` when onnxruntime is not installed.

## Behavior

1. Verify artifact SHA-256 against the manifest via `verify_model_hash`.
2. Verify manifest shapes against the flight contract via `verify_io_contract`.
3. For each golden scene, run inference, measure latency, and compute IoU.
4. Accept when hash, contract, mean IoU, and worst latency all pass thresholds.
5. `onnx_inference_fn` lazily imports onnxruntime and applies sigmoid to logits.

## Errors and faults

Uses `Result` checks from flight verify helpers. `load_manifest` and `onnx_inference_fn`
raise on tooling errors.

## Messages

None.

## Configuration

Callers pass thresholds (`min_iou`, `max_latency_ms`, `iou_threshold`) and expected shapes
explicitly.

## Constraints

- Inference runs through an injected callable. CI tests without onnxruntime.
- IoU is pure NumPy.
- Training lives in a separate model repo. This gate is the intake check in this repo.

## Related documents

- [`tools`](tools.md)
- [`flight.payload.model.verify`](flight/payload/model/verify.md)
