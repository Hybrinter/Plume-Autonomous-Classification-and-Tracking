# flight.payload.inference.segmentor

**Source:** `packages/flight/src/flight/payload/inference/segmentor.py`
**Kind:** module

## Purpose

This module defines a per-tile segmentation interface and scripted and ONNX
implementations. The segmentor emits one probability mask per selected positive
tile. The detector stitches masks into full-frame geometry before blob extraction.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `TileSegmentorBackend` | protocol | `segment_tiles(images, gsd) -> Result[(N, 1, h, w), FaultCode]` |
| `SegmentorBackend` | protocol | Legacy `segment(frame)` wrapper interface |
| `ScriptedSegmentor` | class | Fixed tile probability masks for SIL and unit tests |
| `OnnxSegmentor` | class | Image/GSD ONNX session over logits, then sigmoid |

## Inputs and outputs

`ScriptedSegmentor(prob_mask)` with `prob_mask` shape `(H, W)` float32, or an
exact pre-sliced batch `(N, 1, h, w)`.

`OnnxSegmentor(model_path, expected_sha256, expected_input_shape,
expected_output_shape, expected_gsd_shape=(None, 2))`.

The tile API is `segment_tiles(images, gsd) -> Result[np.ndarray, FaultCode]`,
where `images` is `(N, C, h, w)` and `gsd` is encoded `(N, 2)`. It returns
probability masks `(N, 1, h, w)`.

`ScriptedSegmentor.load_mask(prob_mask)` replaces the stored mask. It is not on
`SegmentorBackend`.

## Behavior

1. `ScriptedSegmentor.segment_tiles` repeats a single mask for the requested
   batch or returns a configured tile batch when the counts match.
2. `ScriptedSegmentor.load_mask` copies a new mask into the stored slot.
3. `OnnxSegmentor.__init__` opens an onnxruntime session through the shared session
   loader. Hash and shape checks are optional.
4. `OnnxSegmentor.segment_tiles` feeds named `image` and `gsd` inputs, supports
   dynamic batches, validates finite logits, and applies a stable sigmoid. The
   exported graph emits logits; sigmoid is not part of the graph.

## Errors and faults

| Fault / error | Trigger |
| --- | --- |
| `INFERENCE_NAN` | Non-finite logits/probabilities or inference runtime failure |
| `FRAME_MALFORMED` | Invalid image or GSD shape/value |
| `ValueError` at init | Hash or I/O contract verification failure |
| `ImportError` at init | onnxruntime not installed |

## Messages

None. The detector consumes the mask in process.

## Configuration

Uses `InferenceConfig.segmentor_model_path` and tile geometry via the composition
root.

## Constraints

onnxruntime loads only when `OnnxSegmentor` is constructed. The module never imports
real or sim HAL drivers. Callers must not overlap `load_mask` with inference. The
slot has no lock.

## Related documents

- [`flight.payload.inference.detector`](detector.md)
- [`flight.payload.inference.classifier`](classifier.md)
- [`flight.payload.blobs`](../blobs.md)
- [`flight.payload.inference.onnx_session`](onnx_session.md)
