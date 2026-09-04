# tools.inference.ort_providers

**Source:** `packages/tools/src/tools/inference/ort_providers.py`
**Kind:** module

## Purpose

This module selects onnxruntime execution providers for accept and bench runs.
The preference order is TensorRT, then CUDA, then CPU. The returned list is the
intersection with providers this runtime has.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `PREFERRED_ORT_PROVIDERS` | constant | Preference order as a tuple of names |
| `resolve_ort_providers` | function | Available providers in that order |

## Inputs and outputs

`resolve_ort_providers() -> list[str]`.

## Behavior

1. Import onnxruntime.
2. Intersect `PREFERRED_ORT_PROVIDERS` with `get_available_providers()`.
3. Return the remaining names in preference order.

## Errors and faults

`ImportError` when onnxruntime is not installed. `RuntimeError` when none of
the preferred providers are available.

## Messages

None.

## Configuration

None. The preference list is a module constant.

## Constraints

Flight session load does not call this helper. CI installs CPU onnxruntime, so
the returned list is `CPUExecutionProvider` alone.

## Related documents

- [`tools.inference.accept`](accept.md)
- [`tools.inference`](../inference.md)
