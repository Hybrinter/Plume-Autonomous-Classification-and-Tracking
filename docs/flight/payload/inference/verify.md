# flight.payload.inference.verify

**Source:** `packages/flight/src/flight/payload/inference/verify.py`
**Kind:** pure module

## Purpose

This module holds pure helpers that verify model artifacts and per-frame inference
timing. The ONNX session loader and `Detector.detect` call them.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `compute_sha256` | function | Returns lowercase hex digest of a file |
| `verify_model_hash` | function | Compares file digest to expected value |
| `verify_io_contract` | function | Compares model tensor shapes to expected shapes |
| `check_inference_latency` | function | Compares elapsed milliseconds to budget |

## Inputs and outputs

`compute_sha256(path)` returns a hex string.

`verify_model_hash(path, expected_sha256)` returns `Result[None, FaultCode]`.

`verify_io_contract(actual_input, actual_output, expected_input, expected_output)`
returns `Result[None, FaultCode]`. `None` dimensions in either shape act as wildcards.

`check_inference_latency(elapsed_ms, budget_ms)` returns `Result[None, FaultCode]`.

## Behavior

1. `verify_model_hash` reads the file bytes and compares SHA-256 to the expected digest.
2. `verify_io_contract` checks equal rank and pairwise dimension match with wildcard
   support for dynamic ONNX dimensions.
3. `check_inference_latency` passes when `budget_ms <= 0` or elapsed time is within
   budget.

## Errors and faults

| Result | Trigger |
| --- | --- |
| `Err(MODEL_CORRUPT)` | Hash mismatch, unreadable file, or shape mismatch |
| `Err(INFERENCE_TIMEOUT)` | Elapsed time exceeds positive budget |

## Messages

None.

## Configuration

Expected hash and shapes come from the model manifest at composition-root load time.
Per-frame budget comes from `InferenceConfig.latency_budget_ms`.

## Constraints

The module does not import onnxruntime. It performs file reads only in `compute_sha256`
and `verify_model_hash`.

## Related documents

- [`flight.payload.inference.onnx_session`](onnx_session.md)
- [`flight.payload.inference.detector`](detector.md)
- [`flight.payload.inference`](../inference.md)
