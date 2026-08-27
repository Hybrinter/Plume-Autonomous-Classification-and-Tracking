# flight.payload.inference.onnx_session

**Source:** `packages/flight/src/flight/payload/inference/onnx_session.py`
**Kind:** module

## Purpose

This module opens onnxruntime sessions for the classifier and segmentor backends.
It verifies an optional artifact hash before load and optional tensor shapes after
load.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `OnnxNamedValue` | protocol | Input or output metadata with `name` and `shape` |
| `OnnxInferenceSession` | protocol | Session `get_inputs`, `get_outputs`, and `run` |
| `onnx_tensor_shape` | function | Maps symbolic dims to `None` |
| `load_onnx_session` | function | Opens a session with optional hash and I/O checks |

## Inputs and outputs

`onnx_tensor_shape(shape) -> tuple[int | None, ...]`.

`load_onnx_session(model_path, expected_sha256, expected_input_shape,
expected_output_shape) -> OnnxInferenceSession`.

## Behavior

1. When `expected_sha256` is set, hash the artifact before constructing a session.
2. Import onnxruntime and open `InferenceSession(model_path)`.
3. When both expected shapes are set, compare session I/O shapes to the contract.

## Errors and faults

| Fault / error | Trigger |
| --- | --- |
| `ValueError` | Hash mismatch or I/O contract mismatch |
| `ImportError` | onnxruntime not installed |

## Messages

None.

## Configuration

None. Callers pass paths and expected shapes.

## Constraints

onnxruntime imports inside `load_onnx_session`. Importing the module does not require
the SDK.

## Related documents

- [`flight.payload.inference.verify`](verify.md)
- [`flight.payload.inference.classifier`](classifier.md)
- [`flight.payload.inference.segmentor`](segmentor.md)
