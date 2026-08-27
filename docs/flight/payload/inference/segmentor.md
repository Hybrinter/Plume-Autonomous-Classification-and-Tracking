# flight.payload.inference.segmentor

**Source:** `packages/flight/src/flight/payload/inference/segmentor.py`
**Kind:** module

## Purpose

This module defines the per-pixel segmentation interface and two implementations.
The segmentor emits a probability mask. Blob extraction runs on that mask after a
positive classifier decision.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SegmentorBackend` | protocol | `segment(frame) -> Result[ndarray, FaultCode]` |
| `ScriptedSegmentor` | class | Fixed probability mask for SIL and unit tests |
| `OnnxSegmentor` | class | ONNX session over logits, then sigmoid |

## Inputs and outputs

`ScriptedSegmentor(prob_mask)` with `prob_mask` shape `(H, W)` float32.

`OnnxSegmentor(model_path, expected_sha256, expected_input_shape,
expected_output_shape)`.

Both implement `segment(ProcessedFrameMsg) -> Result[np.ndarray, FaultCode]` with a
`(H, W)` float32 probability mask.

## Behavior

1. `ScriptedSegmentor.segment` returns the configured mask. It does not read the
   frame tensor.
2. `OnnxSegmentor.__init__` opens an onnxruntime session through the shared session
   loader. Hash and shape checks are optional.
3. `OnnxSegmentor.segment` adds a batch dimension, runs the session, applies
   sigmoid, and returns `probs[0, 0]`. The exported graph emits logits. Sigmoid is
   not part of the graph.

## Errors and faults

| Fault / error | Trigger |
| --- | --- |
| `INFERENCE_NAN` | Non-finite probabilities after sigmoid |
| `ValueError` at init | Hash or I/O contract verification failure |
| `ImportError` at init | onnxruntime not installed |

## Messages

None. The detector consumes the mask in process.

## Configuration

Uses `InferenceConfig.segmentor_model_path` and input geometry via the composition
root.

## Constraints

onnxruntime loads only when `OnnxSegmentor` is constructed. The module never imports
real or sim HAL drivers.

## Related documents

- [`flight.payload.inference.detector`](detector.md)
- [`flight.payload.inference.classifier`](classifier.md)
- [`flight.payload.blobs`](../blobs.md)
- [`flight.payload.inference.onnx_session`](onnx_session.md)
