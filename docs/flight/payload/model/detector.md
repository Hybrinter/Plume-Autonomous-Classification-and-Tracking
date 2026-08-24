# flight.payload.model.detector

**Source:** `packages/flight/src/flight/payload/model/detector.py`
**Kind:** module

## Purpose

This module defines the swappable detector protocol and two backends: a fixed-mask scripted
detector and an ONNX runtime detector.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `DetectorBackend` | protocol | `detect(frame) -> Result[InferenceResultMsg, FaultCode]` |
| `ScriptedDetector` | class | Deterministic mask-backed detector |
| `OnnxDetector` | class | ONNX session-backed detector |

## Inputs and outputs

Both backends implement `detect(frame: ProcessedFrameMsg)`.

`ScriptedDetector(prob_mask, confidence_gate, min_blob_area_px, model_version)`.

`OnnxDetector(model_path, confidence_gate, min_blob_area_px, model_version, expected_sha256,
latency_budget_ms, expected_input_shape, expected_output_shape)`.

## Behavior

**ScriptedDetector**

1. Call `extract_blobs` on the configured fixed mask.
2. Return `Ok(InferenceResultMsg)` with the mask, blobs, zero inference time, and crop metadata
   copied from the frame.

**OnnxDetector**

1. At init, optionally verify SHA-256 and I/O shapes; import `onnxruntime` lazily; open a session.
2. On `detect`, add a batch dimension to the frame tensor and run the session.
3. Apply sigmoid to logits; reject non-finite output.
4. Extract blobs from the probability slice.
5. Check latency against the budget when `latency_budget_ms > 0`.
6. Return `Ok(InferenceResultMsg)` with timing and crop metadata.

## Errors and faults

| Fault | Trigger |
| --- | --- |
| `INFERENCE_NAN` | Non-finite ONNX output |
| `INFERENCE_TIMEOUT` | Elapsed ms exceeds budget |
| Startup | `ImportError` without onnxruntime; `ValueError` on hash or I/O failure |

## Messages

Returns `InferenceResultMsg` for the app shell to publish. Does not publish directly.

## Configuration

Uses detector constructor args sourced from `InferenceConfig` and controller gate fields at the
composition root.

## Constraints

Importing this module does not require `onnxruntime`. Only constructing `OnnxDetector` does.

## Related documents

- [`flight.payload.model`](model.md)
- [`flight.payload.model.blobs`](blobs.md)
- [`flight.payload.model.verify`](verify.md)
