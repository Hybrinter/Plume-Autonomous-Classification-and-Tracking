# tools.ml_models.analysis.capture

**Source:** `packages/tools/src/tools/ml_models/analysis/capture.py`
**Kind:** module
**Status:** stub

## Purpose

This module declares the bounded prediction/evidence persistence boundary
used by the split evaluator.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `CaptureSink` | Protocol | Runtime-checkable sink with a `close` method |

## Inputs and outputs

`CaptureSink.close() -> Result[None, str]`.

## Behavior

This scaffold declares the structural protocol only. Sink implementations
land in the capture phase.

## Errors and faults

`close` returns `Err` with the failure reason.

## Messages

None.

## Configuration

None.

## Constraints

- The protocol is `runtime_checkable`; structural conformance suffices.

## Related documents

- [`tools.ml_models.analysis`](../analysis.md)
- [`tools.ml_models.analysis.evaluate`](evaluate.md)
