# flight.payload.inference.detector

**Source:** `packages/flight/src/flight/payload/inference/detector.py`
**Kind:** module

## Purpose

This module splits each preprocessed full frame into a configured row-major grid,
classifies every tile, segments only positive tiles, stitches the masks, and
extracts blobs. The payload app talks to `DetectorBackend` only. The shared pure
`infer_tiles` helper can also be called by ground adapters.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `DetectorBackend` | protocol | `detect(frame) -> Result[InferenceResultMsg, FaultCode]` |
| `TiledScore` | dataclass | Tile logits, positive flags, and probability masks |
| `infer_tiles` | function | Pure gated classification and segmentation orchestration |
| `Detector` | class | Tiling, inference, mask stitching, and blob extraction |
| `ScriptedDetector` | class | Scripted backends with one-tile test default |
| `OnnxDetector` | class | ONNX backends with an 8 by 8 production default |

## Inputs and outputs

`Detector(classifier, segmentor, confidence_gate, min_blob_area_px, model_version,
latency_budget_ms, grid=(8, 8), gsd_reference_m=15.87, logit_threshold=0.0)`.

`ScriptedDetector(prob_mask, confidence_gate, min_blob_area_px, model_version,
classifier_positive, latency_budget_ms, grid=(1, 1), gsd_reference_m=15.87)`.

`OnnxDetector(segmentor_model_path, classifier_model_path, ..., grid=(8, 8),
gsd_reference_m=15.87, expected_gsd_shape=(None, 2))`.

All implement `detect(ProcessedFrameMsg) -> Result[InferenceResultMsg, FaultCode]`.

`ScriptedDetector.load_mask(prob_mask)` replaces the scripted segmentor mask. It is
not on `DetectorBackend`.

## Behavior

1. `detect` requires finite, positive per-tile GSD in metres with shape `(N, 2)`.
   It encodes each pair as `ln(GSD / gsd_reference_m)` for the model inputs.
2. The classifier runs once over the full tile batch. A tile is positive when its
   logit meets `logit_threshold`; only those tiles are sent to the segmentor.
3. Positive masks scatter into a zero-initialized batch and stitch into a full
   `(H, W)` probability mask. No positive tiles means no segmentor call.
4. The result carries compact per-tile logits, gate flags, and actual GSD in
   metres. Pixel arrays stay local and do not go on the message bus.
5. Wall-clock time covers the full `detect()` call. An exceeded latency budget
   returns `INFERENCE_TIMEOUT`.
6. `ScriptedDetector` defaults to an always-positive classifier and reports
   `inference_ms` as 0.0. Its no-GSD convenience marks synthetic reference GSD
   with `GSD_NOMINAL`; production `Detector` returns `FRAME_MALFORMED` when GSD
   is missing.
7. `OnnxDetector` constructs `OnnxClassifier` and `OnnxSegmentor` at init.
   `ScriptedDetector.load_mask` slices a new full-frame mask into configured tiles.

## Errors and faults

| Fault / error | Trigger |
| --- | --- |
| `INFERENCE_NAN` | Non-finite classifier logit or segmentor probabilities |
| `FRAME_MALFORMED` | Invalid frame geometry, missing GSD, or malformed backend output |
| `INFERENCE_TIMEOUT` | Elapsed time exceeds `latency_budget_ms` when budget is positive |
| `ValueError` at init | Hash or I/O contract verification failure |
| `ImportError` at init | onnxruntime not installed |

## Messages

None directly. The app publishes the returned `InferenceResultMsg`.

## Configuration

Uses `VisionConfig` (`confidence_gate`, `min_blob_area_px`) and `InferenceConfig`
(artifact paths, tile grid, GSD reference, `classifier_logit_threshold`,
`latency_budget_ms`) via the composition root.

## Constraints

onnxruntime loads only when an ONNX backend is constructed. The module never imports
real or sim HAL drivers. Scripted and ONNX paths share `extract_blobs`. Callers must
not overlap `load_mask` with `detect`. The slot has no lock.

## Related documents

- [`flight.payload.inference.classifier`](classifier.md)
- [`flight.payload.inference.segmentor`](segmentor.md)
- [`flight.payload.blobs`](../blobs.md)
- [`flight.payload.app`](../app.md)
