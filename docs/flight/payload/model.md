# flight.payload.model

**Source:** `packages/flight/src/flight/payload/model/`
**Kind:** package

## Purpose

The model package provides swappable detection backends, shared blob extraction, and artifact
verification helpers.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`blobs`](model/blobs.md) | module | Connected-component blob extraction from a probability mask |
| [`detector`](model/detector.md) | module | `DetectorBackend` protocol, scripted and ONNX backends |
| [`verify`](model/verify.md) | module | Hash, I/O shape, and latency checks |

## Package interface

Re-exports: `DetectorBackend`, `ScriptedDetector`, `OnnxDetector`, `extract_blobs`,
`verify_model_hash`, `verify_io_contract`, `check_inference_latency`, `compute_sha256`.

## Interactions

The detector consumes a local `ProcessedFrameMsg` and returns `InferenceResultMsg`. The app
shell publishes the result. `onnxruntime` loads lazily inside `OnnxDetector.__init__`.

## Constraints

Both backends call `extract_blobs` for identical geometry. Startup verification failures in
`OnnxDetector.__init__` raise `ValueError`.

## Related documents

- [`flight.payload`](payload.md)
- [`flight.payload.app`](app.md)
