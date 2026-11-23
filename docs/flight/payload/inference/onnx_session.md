# flight.payload.inference.onnx_session

**Source:** `packages/flight/src/flight/payload/inference/onnx_session.py`
**Kind:** module

## Purpose

This module opens onnxruntime sessions for conditioned classifier and segmentor
backends. It verifies an optional artifact hash before load, then requires two
float32 inputs named `image` and `gsd` and one float32 output. Input declaration
order does not affect validation.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `OnnxNamedValue` | protocol | Input or output metadata with `name`, `shape`, and `type` |
| `OnnxInferenceSession` | protocol | Session `get_inputs`, `get_outputs`, and `run` |
| `onnx_tensor_shape` | function | Maps symbolic dimensions to `None`, rejects malformed dims |
| `load_onnx_session` | function | Opens a session with strict conditioned I/O checks |

## Inputs and outputs

`load_onnx_session(model_path, expected_sha256, expected_input_shape,
expected_output_shape, providers=None, expected_gsd_shape=None) ->
OnnxInferenceSession`.

## Behavior

1. When `expected_sha256` is set, hash the artifact before constructing a session.
2. Import onnxruntime lazily and open `InferenceSession`.
3. Require exactly the `image` and `gsd` inputs plus one output, all float32.
   Input names may appear in either declaration order.
4. Require dynamic batch, image `(None, C, H, W)`, GSD `(None, 2)`, classifier
   output `(None, 1)`, or segmentor output `(None, 1, H, W)`.
5. Supplied expected shapes are compared strictly. Dynamic graph dimensions do
   not wildcard a configured fixed dimension.

## Errors and faults

| Fault / error | Trigger |
| --- | --- |
| `ValueError` | Hash mismatch or conditioned I/O contract mismatch |
| `ImportError` | onnxruntime not installed |

## Messages

None.

## Configuration

Callers may supply expected image, GSD, and output shapes, a hash, and execution
providers.

## Constraints

onnxruntime imports inside `load_onnx_session`. Importing this module does not
require the SDK. Single-input factory artifacts fail the conditioned contract
and must not be relabeled as new models.

## Related documents

- [`flight.payload.inference.contract`](contract.md)
- [`flight.payload.inference.verify`](verify.md)
- [`flight.payload.inference.classifier`](classifier.md)
- [`flight.payload.inference.segmentor`](segmentor.md)
