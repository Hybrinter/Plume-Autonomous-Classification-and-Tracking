# tools.model.accept

**Source:** `packages/tools/src/tools/model/accept.py`
**Kind:** module

## Purpose

The acceptance gate checks a frozen ONNX artifact before it enters `data/models/`.
It runs manifest, hash, I/O contract, golden-scene IoU, and latency checks.

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

`load_manifest(path) -> Manifest`. Raises on missing or malformed JSON.

`compute_iou(pred_mask, gold_mask, threshold=0.5) -> float` in [0, 1]. Two empty
masks score 1.0.

`accept_artifact(...) -> AcceptanceReport`. `accepted` is true only when all
checks pass.

`onnx_inference_fn(artifact_path) -> InferenceFn` maps `(C, H, W)` to a sigmoid
mask `(H, W)`. Raises `ImportError` when onnxruntime is not installed.

## Behavior

1. Verify artifact SHA-256 against the manifest.
2. Verify manifest shapes against the flight contract.
3. For each golden scene, run inference, measure latency, and compute IoU.
4. Accept when hash, contract, mean IoU, and worst latency all pass.
5. `onnx_inference_fn` lazily imports onnxruntime and applies sigmoid to logits.

## Errors and faults

Uses `Result` checks from flight verify helpers. `load_manifest` and
`onnx_inference_fn` raise on tooling errors.

## Messages

None.

## Configuration

Callers pass `min_iou`, `max_latency_ms`, `iou_threshold`, and expected shapes.

## Constraints

Inference runs through an injected callable. CI tests without onnxruntime. IoU is
pure NumPy.

## Related documents

- [`tools.model`](../model.md)
- [`flight.payload.inference.verify`](../../flight/payload/inference/verify.md)
