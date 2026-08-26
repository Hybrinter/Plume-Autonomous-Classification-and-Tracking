# tools.model.export

**Source:** `packages/tools/src/tools/model/export.py`
**Kind:** stub

## Purpose

This module is a scaffold for torch-to-ONNX export and promote into `data/models/`.
`export()` raises `NotImplementedError` in this layer.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `export` | function | Stub. Raises `NotImplementedError`. |

## Inputs and outputs

`export() -> None`. Always raises.

## Behavior

1. Call `export()`.
2. Raise `NotImplementedError`.

## Errors and faults

`NotImplementedError` on every call.

## Messages

None.

## Configuration

None.

## Constraints

This unit is a stub. Importing the module does not import torch. ONNX export is
not implemented.

## Related documents

- [`tools.model`](../model.md)
- [`tools.model.train`](train.md)
- [`tools.model.accept`](accept.md)
