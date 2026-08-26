# flight.payload.model

**Source:** `packages/flight/src/flight/payload/model`
**Kind:** package

## Purpose

The model package provides swappable onboard inference backends. A binary classifier
gates a segmentor. Shared blob extraction turns the probability mask into detections.
Verification helpers check artifact hash, I/O contract, and latency.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`classifier`](model/classifier.md) | module | Presence classifier protocol, scripted and ONNX implementations |
| [`segmentor`](model/segmentor.md) | module | Probability-mask protocol, scripted and ONNX implementations |
| [`detector`](model/detector.md) | module | Composer of classifier, segmentor, and blob extraction |
| [`blobs`](model/blobs.md) | module | Connected-component blob extraction from a probability mask |
| [`onnx_session`](model/onnx_session.md) | module | Lazy onnxruntime session load with hash and shape checks |
| [`verify`](model/verify.md) | module | Hash, I/O contract, and latency verification |

## Package interface

Re-exports: `ClassifierBackend`, `ClassifierDecision`, `Detector`, `DetectorBackend`,
`OnnxClassifier`, `OnnxDetector`, `OnnxSegmentor`, `ScriptedClassifier`,
`ScriptedDetector`, `ScriptedSegmentor`, `SegmentorBackend`, `extract_blobs`,
`check_inference_latency`, `compute_sha256`, `verify_io_contract`, `verify_model_hash`.

## Interactions

The payload app calls `DetectorBackend.detect` and publishes the returned
`InferenceResultMsg`. Verification helpers run at ONNX load time. Latency is checked
per frame inside `Detector.detect`.

## Constraints

`onnxruntime` imports lazily inside session load. Importing the package does not
require the SDK. Hash and shape verification failures at startup raise `ValueError`
in the composition root. A negative classifier skips the segmentor.

## Related documents

- [`flight.payload.app`](../app.md)
- [`flight.core.model_deploy`](../core/model_deploy.md)
