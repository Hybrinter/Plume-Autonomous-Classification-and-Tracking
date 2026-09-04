# tools.inference.export

**Source:** `packages/tools/src/tools/inference/export.py`
**Kind:** module

## Purpose

This module exports a train checkpoint to a frozen ONNX graph and a JSON
manifest. Optional INT8 and FP16 conversion write sibling artifacts.
`quantize_knee` overwrites factory paths with classifier FP16 and segmentor
INT8. `promote` copies a passed artifact into `data/models/`.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `ExportConfig` | class | Frozen export hyperparameters |
| `export` | function | Write ONNX logits plus a Manifest sidecar |
| `reexport_spatial` | function | Rebuild a graph at a new H/W and copy matching weights |
| `convert_fp16` | function | Rewrite an FP32 graph to FP16 with float32 I/O |
| `quantize_int8` | function | Static QDQ INT8 with float32 I/O |
| `quantize_knee` | function | Classifier FP16 and segmentor INT8 in place |
| `write_manifest` | function | Serialize a Manifest as JSON |
| `int8_artifact_path` | function | Sibling ``*.int8.onnx`` path for an FP32 file |
| `fp16_artifact_path` | function | Sibling ``*.fp16.onnx`` path for an FP32 file |
| `promote` | function | Copy a passed artifact to a destination path |
| `GateReport` | protocol | `accepted` and `detail` fields used by promote |

## Inputs and outputs

`export(config) -> (onnx_path, manifest_path, Manifest)` for the FP32 pair.

`reexport_spatial(source_onnx, dest_onnx, *, kind, arch, height, width) ->
(onnx_path, manifest_path, Manifest)`.

`convert_fp16(source_onnx, dest_onnx) -> (onnx_path, manifest_path, Manifest)`.

`quantize_int8(source_onnx, dest_onnx, *, calib_dir="", calib_samples=4) ->
(onnx_path, manifest_path, Manifest)`.

`quantize_knee(classifier_onnx, segmentor_onnx, *, calib_dir="",
calib_samples=4) -> ((cls_onnx, cls_json, cls_manifest), (seg_onnx, seg_json,
seg_manifest))`.

`write_manifest(path, manifest) -> None`.

`int8_artifact_path(fp32_path) -> Path`.

`fp16_artifact_path(fp32_path) -> Path`.

`promote(artifact_path, dest_path, report) -> Path`. Raises `ValueError` when
`report.accepted` is false.

## Behavior

1. Load the checkpoint and rebuild the matching network.
2. Export an ONNX graph named `input` to `logits`. The graph does not include
   sigmoid. When `override_spatial` is true, export uses `ExportConfig` height
   and width even if the checkpoint recorded a different size.
3. Hash the file, write a Manifest sidecar with `quantization` `fp32`.
4. When `int8` is true, run static QDQ PTQ on the FP32 graph. Calibration uses
   the train split of `calib_dir`, or synthetic `[0, 1]` NCHW tensors. Write
   `<stem>.int8.onnx` and `<stem>.int8.json` with `quantization` `int8`. QDQ
   nodes keep graph input and output as float32.
5. When `fp16` is true, convert the FP32 graph to FP16 with float32 I/O and
   write `<stem>.fp16.onnx` plus sidecar `quantization` `fp16`.
6. `reexport_spatial` exports an untrained graph at the new H/W, then copies
   same-name, same-shape ONNX initializers from the source artifact.
7. `quantize_knee` overwrites the classifier path with FP16 and the segmentor
   path with INT8 QDQ. That pair is the factory quality knee.
8. `promote` copies the `.onnx` and sidecar only after a passing gate report.

## Errors and faults

`ImportError` when INT8 or FP16 conversion runs without onnxruntime. `ValueError`
on an unknown kind, a rejected promote, or a calibration geometry mismatch.
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
`(1, 1, H, W)`. Destination names in flight config are
`data/models/active_classifier.onnx` and `data/models/active_segmentor.onnx`.
INT8 and FP16 conversion live behind the `export` extra. Flight `use_int8`
selects an INT8 sibling at driver construction when true. Factory boot leaves
that flag false. The configured files already hold the mixed-knee graphs.
Calibration tensors convert to numpy only for the onnxruntime reader.

## Related documents

- [`tools.inference`](../inference.md)
- [`tools.inference.train`](train.md)
- [`tools.inference.accept`](accept.md)
