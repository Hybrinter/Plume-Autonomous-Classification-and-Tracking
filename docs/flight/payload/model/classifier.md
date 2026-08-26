# flight.payload.model.classifier

**Source:** `packages/flight/src/flight/payload/model/classifier.py`
**Kind:** module

## Purpose

This module defines the binary plume-presence classifier interface and two
implementations. A positive decision allows the segmentor to run. A negative
decision skips segmentation.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `ClassifierDecision` | class | Logit and boolean presence flag |
| `ClassifierBackend` | protocol | `classify(frame) -> Result[ClassifierDecision, FaultCode]` |
| `ScriptedClassifier` | class | Fixed presence decision for SIL and unit tests |
| `OnnxClassifier` | class | ONNX session over a `(1, C, H, W)` to `(1, 1)` logit graph |

## Inputs and outputs

`ScriptedClassifier(positive, logit)`.

`OnnxClassifier(model_path, logit_threshold, expected_sha256, expected_input_shape,
expected_output_shape)`.

Both implement `classify(ProcessedFrameMsg) -> Result[ClassifierDecision, FaultCode]`.
A frame is positive when `logit >= logit_threshold`. The default threshold is `0.0`.

## Behavior

1. `ScriptedClassifier.classify` returns the configured decision. It does not read
   the frame tensor. The default is always-positive.
2. `OnnxClassifier.__init__` opens an onnxruntime session through the shared session
   loader. Hash and shape checks are optional.
3. `OnnxClassifier.classify` adds a batch dimension, runs the session, squeezes the
   logit, and applies the threshold. Non-finite logits raise `INFERENCE_NAN`.

## Errors and faults

| Fault / error | Trigger |
| --- | --- |
| `INFERENCE_NAN` | Non-finite classifier logit |
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

- [`flight.payload.model.detector`](detector.md)
- [`flight.payload.model.segmentor`](segmentor.md)
- [`flight.payload.model.onnx_session`](onnx_session.md)
