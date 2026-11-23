# flight.payload.inference

**Source:** `packages/flight/src/flight/payload/inference`
**Kind:** package

## Purpose

The inference package provides swappable onboard inference backends. A binary classifier
gates the segmentor per tile. The detector stitches the tile probabilities into
a full-frame mask and then extracts blobs.
Verification helpers check artifact hash, I/O contract, and latency.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`classifier`](inference/classifier.md) | module | Presence classifier protocol, scripted and ONNX implementations |
| [`segmentor`](inference/segmentor.md) | module | Probability-mask protocol, scripted and ONNX implementations |
| [`detector`](inference/detector.md) | module | Composer of classifier, segmentor, and blob extraction |
| [`artifact_path`](inference/artifact_path.md) | module | FP32 path to INT8 sibling when `use_int8` is true |
| [`onnx_session`](inference/onnx_session.md) | module | Lazy onnxruntime session load with hash and shape checks |
| [`verify`](inference/verify.md) | module | Hash, I/O contract, and latency verification |
| [`contract`](inference/contract.md) | module | Named image/GSD dynamic-batch graph contract |
| [`runtime`](inference/runtime.md) | module | Lazy session/factory protocols and the verified-session holder |

## Package interface

Re-exports: `ClassifierBackend`, `ClassifierDecision`, `Detector`, `DetectorBackend`,
`InferenceRuntime`, `OnnxClassifier`, `OnnxDetector`, `OnnxRuntimeFactory`,
`OnnxRuntimeSession`, `OnnxSegmentor`, `RuntimeFactory`, `RuntimeSession`,
`ScriptedClassifier`, `ScriptedDetector`, `ScriptedRuntimeFactory`,
`ScriptedRuntimeSession`, `ScriptedSegmentor`, `SegmentorBackend`,
`TileClassifierBackend`, `TileSegmentorBackend`, `TiledScore`, `infer_tiles`,
`check_inference_latency`, `compute_sha256`, `verify_io_contract`, `verify_model_hash`.

## Interactions

The payload app holds an `InferenceRuntime`: the INIT lifecycle loads a session
through the factory, the control owner installs it once verification succeeds,
and the capture path calls `DetectorBackend.detect` on the installed session.
Verification helpers run at ONNX load time inside the factory's `load` call.
Latency is checked per frame inside `Detector.detect`. Geometry and tiling run
before inference; blob extraction lives in `flight.payload.blobs`.

## Constraints

`onnxruntime` imports lazily inside session load. Importing the package does not
require the SDK. Artifact load failures (unreadable files, hash or shape
mismatch, SDK errors) return typed `Err` results from `RuntimeFactory.load`
during the INIT lifecycle; nothing raises in the composition root.
Classifier-negative tiles skip the segmentor during `detect`; `warm_up` always
exercises both models. ONNX
graphs receive float32 `image` and encoded `gsd` inputs with dynamic batch axes.
The production detector defaults to an 8 by 8 grid. The standalone scripted
detector defaults to one tile for small SIL fixtures. The payload app keeps the
sensor frame full-resolution until it is split for inference.
Old single-input factory graphs do not satisfy this contract.

## Related documents

- [`flight.payload.app`](app.md)
- [`flight.payload.blobs`](blobs.md)
- [`flight.core.model_deploy`](../core/model_deploy.md)
