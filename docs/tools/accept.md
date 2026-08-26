# tools.accept

**Source:** `packages/tools/src/tools/accept.py`
**Kind:** module

## Purpose

This module re-exports the acceptance gate from `tools.model.accept`. New code
imports `tools.model.accept`.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `Manifest` | class | Re-export |
| `GoldenScene` | class | Re-export |
| `AcceptanceReport` | class | Re-export |
| `load_manifest` | function | Re-export |
| `compute_iou` | function | Re-export |
| `accept_artifact` | function | Re-export |
| `onnx_inference_fn` | function | Re-export |
| `GoldenClassifierScene` | class | Re-export |
| `ClassifierAcceptanceReport` | class | Re-export |
| `accept_classifier_artifact` | function | Re-export |
| `onnx_classifier_inference_fn` | function | Re-export |

## Inputs and outputs

Same as `tools.model.accept`.

## Behavior

1. Import names from `tools.model.accept`.
2. Expose them at `tools.accept`.

## Errors and faults

Same as `tools.model.accept`.

## Messages

None.

## Configuration

None.

## Constraints

This module has no extra logic. The gate implementation lives in
`tools.model.accept`.

## Related documents

- [`tools.model.accept`](model/accept.md)
- [`tools`](tools.md)
