# flight.payload.model

**Source:** `packages/flight/src/flight/payload/model`
**Kind:** package

## Purpose

The model package provides swappable onboard detection backends, shared blob extraction
from segmentation masks, and pure model-artifact verification helpers.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`detector`](model/detector.md) | module | `DetectorBackend` protocol, scripted and ONNX implementations |
| [`blobs`](model/blobs.md) | module | Connected-component blob extraction from a probability mask |
| [`verify`](model/verify.md) | module | Hash, I/O contract, and latency verification |

## Package interface

Re-exports: `DetectorBackend`, `OnnxDetector`, `ScriptedDetector`, `extract_blobs`,
`check_inference_latency`, `compute_sha256`, `verify_io_contract`, `verify_model_hash`.

## Interactions

Detectors consume a processed frame tensor and return `InferenceResultMsg` via the app
shell, which publishes the message on the bus. Verification helpers run at ONNX load
time and per frame inside `OnnxDetector`.

## Constraints

`onnxruntime` imports lazily inside `OnnxDetector.__init__`; importing the module does
not require the SDK. Both backends share `extract_blobs` for identical detection
geometry. Hash and shape verification failures at startup raise `ValueError` in the
composition root.

## Related documents

- [`flight.payload.app`](../app.md)
- [`flight.core.model_deploy`](../core/model_deploy.md)
