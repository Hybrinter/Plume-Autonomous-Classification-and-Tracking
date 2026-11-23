# flight.payload.inference.classifier

**Source:** `packages/flight/src/flight/payload/inference/classifier.py`
**Kind:** module

## Purpose

This module defines a per-tile binary plume-presence interface and scripted and
ONNX implementations. The detector classifies every row-major tile, then sends
only positive tiles to the segmentor.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `ClassifierDecision` | class | Logit and boolean presence flag |
| `TileClassifierBackend` | protocol | `classify_tiles(images, gsd) -> Result[(N,), FaultCode]` logits |
| `ClassifierBackend` | protocol | Legacy `classify(frame)` wrapper interface |
| `ScriptedClassifier` | class | Fixed per-tile logits for SIL and unit tests |
| `OnnxClassifier` | class | ONNX image/GSD session returning `(N, 1)` logits |

## Inputs and outputs

`ScriptedClassifier(positive, logit)`.

`OnnxClassifier(model_path, logit_threshold, expected_sha256, expected_input_shape,
expected_output_shape, expected_gsd_shape=(None, 2))`.

The tile API is `classify_tiles(images, gsd) -> Result[np.ndarray, FaultCode]`,
where `images` is finite NCHW float data and `gsd` is finite encoded shape `(N, 2)`.
It returns raw float logits of shape `(N,)`. `logit_threshold` is applied by the
detector, with the default threshold `0.0`. The legacy `classify(frame)` wrapper
accepts one image only.

## Behavior

1. `ScriptedClassifier.classify_tiles` returns one fixed configured logit per
   tile. Its legacy wrapper returns the configured boolean decision.
2. `OnnxClassifier.__init__` opens an onnxruntime session through the shared session
   loader. Hash and shape checks are optional.
3. `OnnxClassifier.classify_tiles` feeds named `image` and `gsd` inputs, supports
   dynamic batches, and returns raw logits. It returns `INFERENCE_NAN` for
   non-finite logits or an inference runtime failure.

## Errors and faults

| Fault / error | Trigger |
| --- | --- |
| `INFERENCE_NAN` | Non-finite logits or inference runtime failure |
| `FRAME_MALFORMED` | Invalid tile or GSD shape/value |
| `ValueError` at init | Hash or I/O contract verification failure |
| `ImportError` at init | onnxruntime not installed |

## Messages

None. The detector consumes the decision in process.

## Configuration

Uses `InferenceConfig.classifier_model_path` and
`InferenceConfig.classifier_logit_threshold` via the composition root.

## Constraints

onnxruntime loads only when `OnnxClassifier` is constructed. The module never imports
real or sim HAL drivers.

## Related documents

- [`flight.payload.inference.detector`](detector.md)
- [`flight.payload.inference.segmentor`](segmentor.md)
- [`flight.payload.inference.onnx_session`](onnx_session.md)
