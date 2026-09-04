# flight.payload.inference.artifact_path

**Source:** `packages/flight/src/flight/payload/inference/artifact_path.py`
**Kind:** pure module

## Purpose

This module maps a configured FP32 ONNX path to the INT8 sibling path when
`use_int8` is true. The sibling name is `<stem>.int8.onnx` next to the FP32 file.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `int8_sibling_path` | function | Sibling ``*.int8.onnx`` path for an FP32 file |
| `resolve_quantized_path` | function | FP32 path, or the INT8 sibling when requested |

## Inputs and outputs

`int8_sibling_path(fp32_path) -> Path`.

`resolve_quantized_path(fp32_path, use_int8) -> str`. Returns `fp32_path` when
`use_int8` is false.

## Behavior

1. `int8_sibling_path` replaces the file name with `<stem>.int8.onnx`.
2. `resolve_quantized_path` returns the FP32 path when `use_int8` is false.
3. `resolve_quantized_path` returns the sibling path when `use_int8` is true.

## Errors and faults

None. Missing files are reported later by session load.

## Messages

None.

## Configuration

Callers pass `inference.classifier_model_path` or
`inference.segmentor_model_path` and `inference.use_int8`.

## Constraints

The helper does not open files. Factory boot writes the mixed-knee graphs
into the configured paths (classifier FP16, segmentor INT8). `use_int8`
stays false.

## Related documents

- [`flight.core.select_drivers`](../../core/select_drivers.md)
- [`flight.payload.inference.onnx_session`](onnx_session.md)
