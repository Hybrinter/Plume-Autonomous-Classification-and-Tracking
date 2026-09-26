# tools.ml_models.export.onnx

**Source:** `packages/tools/src/tools/ml_models/export/onnx.py`
**Kind:** module

## Purpose

This module exports a train checkpoint to a frozen ONNX graph and a JSON
manifest. Optional INT8 and FP16 conversion write sibling artifacts.
`quantize_knee` overwrites factory paths with classifier FP16 and segmentor
INT8. `promote` copies a passed artifact. Graphs emit logits. They do not
include sigmoid.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `ExportConfig` | class | Frozen export hyperparameters |
| `export` | function | Write ONNX logits plus a Manifest sidecar |
| `reexport_spatial` | function | Rebuild a graph at a new H/W; a same-name shape mismatch is an error |
| `convert_fp16` | function | Rewrite an FP32 graph to FP16 with float32 I/O |
| `quantize_int8` | function | Static QDQ INT8 with float32 I/O |
| `quantize_knee` | function | Classifier FP16 and segmentor INT8 in place |
| `write_manifest` | function | Serialize a Manifest as JSON |
| `int8_artifact_path` | function | Sibling `*.int8.onnx` path for an FP32 file |
| `fp16_artifact_path` | function | Sibling `*.fp16.onnx` path for an FP32 file |
| `promote` | function | Copy a passed artifact to a destination path |
| `resolve_export_hw` | function | Choose the traced height and width |
| `GateReport` | protocol | `accepted` and `detail` fields used by promote |

## Inputs and outputs

`export(config) -> (onnx_path, manifest_path, Manifest)` for the FP32 pair.

`resolve_export_hw(...) -> (height, width)`.

`reexport_spatial(source_onnx, dest_onnx, *, kind, arch, height, width) ->
(onnx_path, manifest_path, Manifest)`.

`convert_fp16(source_onnx, dest_onnx) -> (onnx_path, manifest_path, Manifest)`.

`quantize_int8(source_onnx, dest_onnx, *, calib_dir="", calib_samples=4) ->
(onnx_path, manifest_path, Manifest)`.

`quantize_knee(classifier_onnx, segmentor_onnx, *, calib_dir="",
calib_samples=4) -> ((cls_onnx, cls_json, cls_manifest), (seg_onnx, seg_json,
seg_manifest))`.

`write_manifest(path, manifest) -> None`.

`promote(artifact_path, dest_path, report) -> Path`. Raises `ValueError` when
`report.accepted` is false.

## Behavior

1. Load the checkpoint and rebuild the matching network.
2. Choose the trace size with `resolve_export_hw`. A flight-promotable run
   traces `(1, C, 1544, 2064)` with today's `InferenceConfig` defaults.
   `C` is the checkpoint channel count. A research run traces the checkpoint
   height and width. `override_spatial` uses the `ExportConfig` size.
3. Export an ONNX graph named `input` to `logits`. The graph does not include
   sigmoid.
4. Hash the file and write a Manifest sidecar with `quantization` `fp32`.
   The sidecar adds `ingest_path` and `radiometry` when the checkpoint stores
   them.
5. When `int8` is true, run static QDQ PTQ and write `<stem>.int8.onnx`.
   Graph input and output stay float32.
6. When `fp16` is true, convert the FP32 graph to FP16 with float32 I/O.
7. `reexport_spatial` exports an untrained graph at the new H/W, then copies
   same-name, same-shape initializers from the source artifact. A same-name
   initializer with a different shape raises `ValueError`. A destination name
   with no source counterpart is skipped.
8. `promote` copies the `.onnx` and sidecar after a passing gate report.

## Errors and faults

`ImportError` when INT8 or FP16 conversion runs without onnxruntime.
`ValueError` on an unknown kind, a rejected promote, a calibration geometry
mismatch, or a same-name ONNX initializer whose shapes differ.
`FileNotFoundError` on a missing checkpoint, source ONNX, sidecar, or
calibration pack.

## Messages

None.

## Configuration

`ExportConfig` carries kind, checkpoint path, output path, geometry, version,
repo SHA, dataset hash, ONNX opset (default 17), `int8`, `fp16`, `calib_dir`,
`calib_samples` (default 4), and `override_spatial` (default false).

## Constraints

Torch imports at module level. onnxruntime imports inside the INT8 and FP16
paths. Classifier output shape is `(1, 1)`. Segmentor output shape is
`(1, 1, H, W)`. A flight-promotable trace does not require
`override_spatial`.

## Related documents

- [`tools.ml_models.export`](../export.md)
- [`tools.ml_models.export.pair`](pair.md)
- [`tools.ml_models.export.accept`](accept.md)
- [`flight.payload.inference.verify`](../../../flight/payload/inference/verify.md)
