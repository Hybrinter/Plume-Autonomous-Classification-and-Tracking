# tools.inference.export

**Source:** `packages/tools/src/tools/inference/export.py`
**Kind:** module

## Purpose

This module exports a train checkpoint to a frozen ONNX graph and a JSON
manifest. Optional INT8 export writes a sibling QDQ artifact. `promote` copies
a passed artifact into `data/models/`.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `ExportConfig` | class | Frozen export hyperparameters |
| `export` | function | Write ONNX logits plus a Manifest sidecar |
| `write_manifest` | function | Serialize a Manifest as JSON |
| `int8_artifact_path` | function | Sibling ``*.int8.onnx`` path for an FP32 file |
| `promote` | function | Copy a passed artifact to a destination path |
| `GateReport` | protocol | `accepted` and `detail` fields used by promote |

## Inputs and outputs

`export(config) -> (onnx_path, manifest_path, Manifest)` for the FP32 pair.

`write_manifest(path, manifest) -> None`.

`int8_artifact_path(fp32_path) -> Path`.

`promote(artifact_path, dest_path, report) -> Path`. Raises `ValueError` when
`report.accepted` is false.

## Behavior

1. Load the checkpoint and rebuild the matching network.
2. Export an ONNX graph named `input` to `logits`. The graph does not include
   sigmoid.
3. Hash the file, write a Manifest sidecar with `quantization` `fp32`.
4. When `int8` is true, run static QDQ PTQ on the FP32 graph. Calibration uses
   the train split of `calib_dir`, or synthetic `[0, 1]` NCHW tensors. Write
   `<stem>.int8.onnx` and `<stem>.int8.json` with `quantization` `int8`. QDQ
   nodes keep graph input and output as float32.
5. `promote` copies the `.onnx` and sidecar only after a passing gate report.

## Errors and faults

`ImportError` when torch is not installed, or when INT8 export runs without
onnxruntime. `ValueError` on an unknown kind, a rejected promote, or a
calibration geometry mismatch. `FileNotFoundError` on a missing checkpoint or
calibration pack.

## Messages

None.

## Configuration

`ExportConfig` carries kind, checkpoint path, output path, geometry, version,
repo SHA, dataset hash, ONNX opset (default 17), `int8`, `calib_dir`, and
`calib_samples` (default 4).

## Constraints

Importing the module does not import torch or onnxruntime. Classifier output
shape is `(1, 1)`. Segmentor output shape is `(1, 1, H, W)`. Destination names
in flight config are `data/models/active_classifier.onnx` and
`data/models/active_segmentor.onnx`. INT8 export lives behind the `export`
extra. Flight `use_int8` does not select the INT8 file.

## Related documents

- [`tools.inference`](../inference.md)
- [`tools.inference.train`](train.md)
- [`tools.inference.accept`](accept.md)
