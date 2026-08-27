# flight.payload.inference.detector

**Source:** `packages/flight/src/flight/payload/inference/detector.py`
**Kind:** module

## Purpose

This module composes the classifier, segmentor, and blob extraction into one
`detect()` call. The payload app talks to `DetectorBackend` only. The class holds
pipeline knobs (blob thresholds, version string, latency budget). It does not hold
per-frame network state.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `DetectorBackend` | protocol | `detect(frame) -> Result[InferenceResultMsg, FaultCode]` |
| `Detector` | class | Classifier then segmentor then `extract_blobs` |
| `ScriptedDetector` | class | Scripted classifier plus scripted segmentor |
| `OnnxDetector` | class | ONNX classifier plus ONNX segmentor |

## Inputs and outputs

`Detector(classifier, segmentor, confidence_gate, min_blob_area_px, model_version,
latency_budget_ms)`.

`ScriptedDetector(prob_mask, confidence_gate, min_blob_area_px, model_version,
classifier_positive, latency_budget_ms)`.

`OnnxDetector(segmentor_model_path, classifier_model_path, ...)`.

All implement `detect(ProcessedFrameMsg) -> Result[InferenceResultMsg, FaultCode]`.

## Behavior

1. `detect` runs the classifier. A negative decision returns a zero mask, an empty
   blob list, and does not call the segmentor.
2. A positive decision runs the segmentor, then `extract_blobs`.
3. Wall-clock time covers the full `detect()` call. A positive `latency_budget_ms`
   raises `INFERENCE_TIMEOUT` when exceeded.
4. `ScriptedDetector` defaults to an always-positive classifier and reports
   `inference_ms` as 0.0.
5. `OnnxDetector` constructs `OnnxClassifier` and `OnnxSegmentor` at init.

## Errors and faults

| Fault / error | Trigger |
| --- | --- |
| `INFERENCE_NAN` | Non-finite classifier logit or segmentor probabilities |
| `INFERENCE_TIMEOUT` | Elapsed time exceeds `latency_budget_ms` when budget is positive |
| `ValueError` at init | Hash or I/O contract verification failure |
| `ImportError` at init | onnxruntime not installed |

## Messages

None directly. The app publishes the returned `InferenceResultMsg`.

## Configuration

Uses `ControllerConfig` (`confidence_gate`, `min_blob_area_px`) and `InferenceConfig`
(artifact paths, `classifier_logit_threshold`, `latency_budget_ms`) via the
composition root.

## Constraints

onnxruntime loads only when an ONNX backend is constructed. The module never imports
real or sim HAL drivers. Scripted and ONNX paths share `extract_blobs`.

## Related documents

- [`flight.payload.inference.classifier`](classifier.md)
- [`flight.payload.inference.segmentor`](segmentor.md)
- [`flight.payload.blobs`](../blobs.md)
- [`flight.payload.app`](../app.md)
