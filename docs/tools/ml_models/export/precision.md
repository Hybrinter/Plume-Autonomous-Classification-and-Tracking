# tools.ml_models.export.precision

**Source:** `packages/tools/src/tools/ml_models/export/precision.py`
**Kind:** module

## Purpose

This module converts a validated FP32 two-input artifact to FP16 graph
weights or static QDQ INT8. Both conversions keep float32 I/O, validate
the source hash and graph before converting, validate the converted
graph, derive the sidecar from the base manifest, and publish both files
atomically without overwriting existing outputs.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `convert_fp16` | function | `Result[Path, str]` FP16 conversion with float32 I/O |
| `quantize_int8` | function | `Result[Path, str]` static QDQ INT8 quantization |

## Inputs and outputs

`convert_fp16(source, dest)` and `quantize_int8(source, dest, *,
dataset, calib_samples=32)` take `.onnx` paths and return `Result` of
the destination path. `dataset` is the one finished dataset directory
that supplies INT8 calibration rows.

## Behavior

1. The source and its sidecar must exist; `dest` and its sidecar must be
   new and different from the source.
2. The source sidecar loads as a `ModelManifest` and `open_session`
   validates the artifact hash and declared graph.
3. FP16 rewrites weights through `convert_float_to_float16` with
   `keep_io_types=True`. INT8 calls `quantize_static` with
   `QuantFormat.QDQ` and `QInt8` activations and weights over
   `calibration_batches` pairs fed by a `CalibrationDataReader`.
4. The derived manifest only changes `sha256`, `version`, and
   `quantization`; shapes and `input_types`/`output_type` stay float32.
5. The converted artifact is validated through `open_session` and both
   files publish through exclusive sibling links; the temporary survives
   a sidecar-publish failure so rollback removes only the artifact this
   call linked.

## Errors and faults

Missing source or sidecar, existing outputs, equal source/dest paths,
validation failures, calibration failures, and SDK errors return `Err`.

## Messages

None.

## Configuration

`calib_samples` bounds INT8 calibration batches (default 32).

## Constraints

- Destinations are never overwritten and never equal the source.
- onnx and onnxruntime import lazily; a missing SDK returns `Err`.
- A converted artifact needs fresh acceptance evidence; its new SHA-256
  does not match the source report.
- No in-place `quantize_knee` helper exists.

## Related documents

- [`tools.ml_models.export`](../export.md)
- [`tools.ml_models.export.calibration`](calibration.md)
- [`tools.ml_models.export.session`](session.md)
- [`tools.ml_models.cli`](../cli.md)
