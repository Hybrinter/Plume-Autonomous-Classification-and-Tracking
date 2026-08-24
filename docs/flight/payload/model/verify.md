# flight.payload.model.verify

**Source:** `packages/flight/src/flight/payload/model/verify.py`
**Kind:** pure module

## Purpose

This module holds pure helpers to verify model artifacts at load time and per-frame latency at
runtime.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `compute_sha256` | function | Returns hex digest of a file |
| `verify_model_hash` | function | Compares digest to expected value |
| `verify_io_contract` | function | Compares input and output shapes |
| `check_inference_latency` | function | Compares elapsed ms to budget |

## Inputs and outputs

`compute_sha256(path)` returns a lowercase hex string.

`verify_model_hash(path, expected_sha256)` returns `Result[None, FaultCode]`.

`verify_io_contract(actual_input, actual_output, expected_input, expected_output)` returns
`Result[None, FaultCode]`. `None` dimensions match any size.

`check_inference_latency(elapsed_ms, budget_ms)` returns `Result[None, FaultCode]`. A budget
`<= 0` disables the check.

## Behavior

1. Hash verification reads file bytes and compares SHA-256.
2. Shape matching requires equal rank; each dimension matches when equal or when either side is
   `None`.
3. Latency check fails when `budget_ms > 0` and elapsed time exceeds the budget.

## Errors and faults

| Result | Trigger |
| --- | --- |
| `Err(MODEL_CORRUPT)` | Unreadable file, hash mismatch, shape mismatch |
| `Err(INFERENCE_TIMEOUT)` | Elapsed ms above budget |

## Messages

None.

## Configuration

Expected hash and shapes come from deployment metadata at detector construction.
`latency_budget_ms` comes from `InferenceConfig`.

## Constraints

Pure module aside from file reads in hash helpers. No `onnxruntime` import.

## Related documents

- [`flight.payload.model`](model.md)
- [`flight.payload.model.detector`](detector.md)
