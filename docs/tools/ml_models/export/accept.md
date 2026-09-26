# tools.ml_models.export.accept

**Source:** `packages/tools/src/tools/ml_models/export/accept.py`
**Kind:** module

## Purpose

The acceptance gate checks a frozen ONNX artifact. It runs manifest, hash,
I/O contract, golden-scene quality, and latency checks. The manifest records
`quantization`. It also records `ingest_path` and `radiometry` when the
sidecar has those keys.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `Manifest` | class | Sidecar JSON fields |
| `GoldenScene` | class | Input tensor and expected mask |
| `AcceptanceReport` | class | Per-check booleans and the accept flag |
| `load_manifest` | function | Parse manifest JSON |
| `compute_iou` | function | Re-export of mask IoU |
| `accept_artifact` | function | Run the segmentor gate |
| `onnx_inference_fn` | function | onnxruntime-backed mask callable |
| `GoldenClassifierScene` | class | Input tensor and presence label |
| `ClassifierAcceptanceReport` | class | Hash, contract, accuracy, and latency |
| `accept_classifier_artifact` | function | Classifier gate with binary accuracy |
| `accept_kind` | function | Dispatch on classifier or segmentor |
| `onnx_classifier_inference_fn` | function | onnxruntime callable that returns a logit |
| `load_golden_scenes` | function | Segmentor scenes from a pack split |
| `load_golden_classifier_scenes` | function | Classifier scenes from a pack split |

## Inputs and outputs

`load_manifest(path) -> Manifest`. Missing `quantization` defaults to `fp32`.
Missing `ingest_path` and `radiometry` default to empty strings.

`accept_artifact(...) -> AcceptanceReport`. `accepted` is true only when hash,
contract, IoU, and latency all pass.

`accept_classifier_artifact(...) -> ClassifierAcceptanceReport`.

`accept_kind(..., flight=False) -> AcceptanceReport | ClassifierAcceptanceReport`.
When `flight` is true, expected shapes come from `InferenceConfig`: input
`(1, len(input_bands), H, W)`, classifier output `(1, 1)`, and segmentor
output `(1, 1, H, W)`. When `flight` is false, the caller supplies
`expected_input`, `height`, and `width`.

`load_golden_scenes(pack_dir, split="test", limit=0) -> list[GoldenScene]`.

`load_golden_classifier_scenes(pack_dir, split="test", limit=0) ->
list[GoldenClassifierScene]`.

## Behavior

1. Verify artifact SHA-256 against the manifest.
2. Verify manifest shapes against the expected input and output.
3. For each golden scene, run inference, measure latency, and score quality.
4. Accept when hash, contract, quality, and worst latency all pass.
5. The report also stores median and 95th-percentile latency.
6. `iou_ok` and `accuracy_ok` require a non-empty scene list.
7. `onnx_inference_fn` applies sigmoid to logits. The classifier callable
   returns the logit.
8. `accept_kind` with `flight` true replaces the caller's shapes with
   `InferenceConfig`.

## Errors and faults

`load_manifest` raises on a missing or malformed sidecar. Unknown `kind`
raises `ValueError`. A missing onnxruntime raises `ImportError` from the live
callables.

## Messages

None.

## Configuration

Callers pass `min_iou`, `min_accuracy`, `max_latency_ms`, and expected shapes.
`flight` selects the `InferenceConfig` contract. Live sessions call
`resolve_ort_providers`.

## Constraints

Golden scenes carry torch tensors. `compute_iou` comes from
`tools.ml_models.train.metrics`. The live path converts tensors to numpy for
onnxruntime.

## Related documents

- [`tools.ml_models.export`](../export.md)
- [`tools.ml_models.export.ort_providers`](ort_providers.md)
- [`tools.ml_models.train.metrics`](../train/metrics.md)
- [`flight.payload.inference.verify`](../../../flight/payload/inference/verify.md)
