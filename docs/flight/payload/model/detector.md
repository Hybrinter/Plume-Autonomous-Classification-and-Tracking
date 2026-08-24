# flight.payload.model.detector

**Source:** `packages/flight/src/flight/payload/model/detector.py`
**Kind:** module

## Purpose

This module defines the swappable detection backend interface and two implementations.
SIL and tests use a deterministic scripted mask; flight uses a frozen ONNX model via
onnxruntime.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `DetectorBackend` | protocol | `detect(frame) -> Result[InferenceResultMsg, FaultCode]` |
| `ScriptedDetector` | class | Fixed probability mask for SIL and unit tests |
| `OnnxDetector` | class | ONNX session with sigmoid, threshold, and latency check |

## Inputs and outputs

`ScriptedDetector(prob_mask, confidence_gate, min_blob_area_px, model_version)`.

`OnnxDetector(model_path, confidence_gate, min_blob_area_px, model_version,
expected_sha256, latency_budget_ms, expected_input_shape, expected_output_shape)`.

Both implement `detect(ProcessedFrameMsg) -> Result[InferenceResultMsg, FaultCode]`.

## Behavior

1. `ScriptedDetector.detect` runs `extract_blobs` on the configured mask and builds an
   `InferenceResultMsg` with zero inference time.
2. `OnnxDetector.__init__` optionally verifies the artifact hash, imports onnxruntime,
   opens the session, and optionally verifies input/output tensor shapes.
3. `OnnxDetector.detect` adds a batch dimension, runs the session, applies sigmoid,
   checks for non-finite output, extracts blobs, measures wall time, and checks the
   latency budget.
4. Both backends copy `crop_origin_px` and `scale_factor` from the processed frame into
   the inference result.

## Errors and faults

| Fault / error | Trigger |
| --- | --- |
| `INFERENCE_NAN` | Non-finite probabilities after sigmoid (ONNX) |
| `INFERENCE_TIMEOUT` | Elapsed time exceeds `latency_budget_ms` when budget is positive |
| `ValueError` at init | Hash or I/O contract verification failure |
| `ImportError` at init | onnxruntime not installed |

## Messages

None directly. The app publishes the returned `InferenceResultMsg`.

## Configuration

Uses detection thresholds from `ControllerConfig` (`confidence_gate`,
`min_blob_area_px`) and `InferenceConfig` (`latency_budget_ms`, model path and shapes
via composition root).

## Constraints

onnxruntime loads only when `OnnxDetector` is constructed. The module never imports
real or sim HAL drivers.

## Related documents

- [`flight.payload.model.blobs`](blobs.md)
- [`flight.payload.model.verify`](verify.md)
- [`flight.payload.app`](../app.md)
