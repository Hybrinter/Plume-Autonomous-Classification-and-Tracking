# tools.model.export

**Source:** `packages/tools/src/tools/model/export.py`
**Kind:** module

## Purpose

This module exports a train checkpoint to a frozen ONNX graph and a JSON
manifest. `promote` copies a passed artifact into `data/models/`.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `ExportConfig` | class | Frozen export hyperparameters |
| `export` | function | Write ONNX logits plus a Manifest sidecar |
| `write_manifest` | function | Serialize a Manifest as JSON |
| `promote` | function | Copy a passed artifact to a destination path |
| `GateReport` | protocol | `accepted` and `detail` fields used by promote |

## Inputs and outputs

`export(config) -> (onnx_path, manifest_path, Manifest)`.

`write_manifest(path, manifest) -> None`.

`promote(artifact_path, dest_path, report) -> Path`. Raises `ValueError` when
`report.accepted` is false.

## Behavior

1. Load the checkpoint and rebuild the matching network.
2. Export an ONNX graph named `input` to `logits`. The graph does not include
   sigmoid.
3. Hash the file, write a Manifest sidecar with the same stem and `.json`.
4. `promote` copies the `.onnx` and sidecar only after a passing gate report.

## Errors and faults

`ImportError` when torch is not installed. `ValueError` on an unknown kind or a
rejected promote. `FileNotFoundError` on a missing checkpoint.

## Messages

None.

## Configuration

`ExportConfig` carries kind, checkpoint path, output path, geometry, version,
repo SHA, dataset hash, and ONNX opset (default 17).

## Constraints

Importing the module does not import torch. Classifier output shape is `(1, 1)`.
Segmentor output shape is `(1, 1, H, W)`. Destination names in flight config are
`data/models/active_classifier.onnx` and `data/models/active_segmentor.onnx`.

## Related documents

- [`tools.model`](../model.md)
- [`tools.model.train`](train.md)
- [`tools.model.accept`](accept.md)
