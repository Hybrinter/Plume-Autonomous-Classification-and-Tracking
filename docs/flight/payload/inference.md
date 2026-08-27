# flight.payload.inference

**Source:** `packages/flight/src/flight/payload/inference`
**Kind:** package

## Purpose

The inference package provides swappable onboard inference backends. A binary classifier
gates a segmentor. The detector composer runs both networks and then blob extraction.
Verification helpers check artifact hash, I/O contract, and latency.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`classifier`](inference/classifier.md) | module | Presence classifier protocol, scripted and ONNX implementations |
| [`segmentor`](inference/segmentor.md) | module | Probability-mask protocol, scripted and ONNX implementations |
| [`detector`](inference/detector.md) | module | Composer of classifier, segmentor, and blob extraction |
| [`onnx_session`](inference/onnx_session.md) | module | Lazy onnxruntime session load with hash and shape checks |
| [`verify`](inference/verify.md) | module | Hash, I/O contract, and latency verification |

## Package interface

Re-exports: `ClassifierBackend`, `ClassifierDecision`, `Detector`, `DetectorBackend`,
`OnnxClassifier`, `OnnxDetector`, `OnnxSegmentor`, `ScriptedClassifier`,
`ScriptedDetector`, `ScriptedSegmentor`, `SegmentorBackend`,
`check_inference_latency`, `compute_sha256`, `verify_io_contract`, `verify_model_hash`.

## Interactions

The payload app calls `DetectorBackend.detect` and publishes the returned
`InferenceResultMsg`. Verification helpers run at ONNX load time. Latency is checked
per frame inside `Detector.detect`. Blob extraction lives in `flight.payload.blobs`.

## Constraints

`onnxruntime` imports lazily inside session load. Importing the package does not
require the SDK. Hash and shape verification failures at startup raise `ValueError`
in the composition root. A negative classifier skips the segmentor.

## Related documents

- [`flight.payload.app`](app.md)
- [`flight.payload.blobs`](blobs.md)
- [`flight.core.model_deploy`](../core/model_deploy.md)
