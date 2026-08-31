# tools.inference.accept

**Source:** `packages/tools/src/tools/inference/accept.py`
**Kind:** module

## Purpose

The acceptance gate checks a frozen ONNX artifact before it enters `data/models/`.
It runs manifest, hash, I/O contract, golden-scene IoU, and latency checks.
The manifest records `quantization` (`fp32` or `int8`).

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `Manifest` | class | Sidecar JSON fields, including `quantization` |
| `GoldenScene` | class | Input tensor and expected mask for IoU scoring |
| `AcceptanceReport` | class | Per-check booleans and aggregate accept flag |
| `load_manifest` | function | Parse manifest JSON |
| `compute_iou` | function | Re-export of mask IoU from `tools.inference.metrics` |
| `accept_artifact` | function | Run the segmentor gate and return a report |
| `onnx_inference_fn` | function | Build an onnxruntime-backed mask callable |
| `GoldenClassifierScene` | class | Input tensor and presence label |
| `ClassifierAcceptanceReport` | class | Hash, contract, accuracy, latency, accept flag |
| `accept_classifier_artifact` | function | Classifier gate with binary accuracy |
| `accept_kind` | function | Kind dispatch to the classifier or segmentor gate |
| `onnx_classifier_inference_fn` | function | onnxruntime callable that returns a logit |
| `load_golden_scenes` | function | Segmentor golden scenes from a processed pack split |
| `load_golden_classifier_scenes` | function | Classifier golden scenes from a processed pack split |

## Inputs and outputs

`load_manifest(path) -> Manifest`. Raises on missing or malformed JSON.
Missing `quantization` defaults to `fp32`.

`compute_iou(pred_mask, gold_mask, threshold=0.5) -> float` in [0, 1]. Two empty
masks score 1.0.

`accept_artifact(...) -> AcceptanceReport`. `accepted` is true only when all
checks pass.

`onnx_inference_fn(artifact_path) -> InferenceFn` maps `(C, H, W)` torch
tensors to a sigmoid mask `(H, W)`. Raises `ImportError` when onnxruntime is
not installed. The onnxruntime session consumes numpy arrays.

`accept_classifier_artifact(...) -> ClassifierAcceptanceReport`. A frame is
positive when logit >= `logit_threshold` (default 0.0).

`accept_kind(kind, ...) -> AcceptanceReport | ClassifierAcceptanceReport`.
Raises `ValueError` on an unknown kind.

`onnx_classifier_inference_fn(artifact_path) -> ClassifierInferenceFn` maps
`(C, H, W)` to a scalar logit.

`load_golden_scenes(pack_dir, split="test", limit=0) -> list[GoldenScene]`.
Reads a processed pack, selects the named split (`train`, `val`, or `test`),
and returns one `GoldenScene` per sample. Each scene holds a cloned `(C, H, W)`
input tensor and an `(H, W)` reference mask. `limit` caps the count; zero takes
the whole split. Raises `ValueError` on an unknown split name.

`load_golden_classifier_scenes(pack_dir, split="test", limit=0) ->
list[GoldenClassifierScene]`. Same split and limit rules. Each scene holds a
cloned `(C, H, W)` input and a `label_positive` bool (`labels[index, 0] >= 0.5`).

## Behavior

1. Verify artifact SHA-256 against the manifest.
2. Verify manifest shapes against the flight contract.
3. For each golden scene, run inference, measure latency, and compute IoU.
4. Accept when hash, contract, mean IoU, and worst latency all pass.
5. The report also stores median and 95th-percentile latency. The gate does not
   use them. They describe the same run without the outlier of one slow scene.
6. `iou_ok` and `accuracy_ok` require a non-empty scene list.
7. `onnx_inference_fn` lazily imports onnxruntime and applies sigmoid to logits.
8. `accept_classifier_artifact` uses binary accuracy in place of mask IoU.
9. `accept_kind` matches on `kind` and calls the matching gate with live ONNX
   inference. The CLI `accept` command and finalize `_accept` both call it.
10. `load_golden_scenes` and `load_golden_classifier_scenes` clone one sample at
    a time from the pack. Classifier scenes skip `masks.npy`.

## Errors and faults

Uses `Result` checks from flight verify helpers. `load_manifest`,
`load_golden_scenes`, `load_golden_classifier_scenes`, and `onnx_inference_fn`
raise on tooling errors.

## Messages

None.

## Configuration

Callers pass `min_iou`, `min_accuracy`, `max_latency_ms`, `iou_threshold`,
`logit_threshold`, and expected shapes.

## Constraints

Golden scenes carry torch tensors. Injected test callables take tensors.
`onnx_inference_fn` converts to numpy for onnxruntime. `compute_iou` comes from
`tools.inference.metrics`. Classifier accuracy uses the same metrics module.

## Related documents

- [`tools.inference`](../inference.md)
- [`tools.inference.metrics`](metrics.md)
- [`flight.payload.inference.verify`](../../flight/payload/inference/verify.md)
