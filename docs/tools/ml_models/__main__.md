# tools.ml_models.__main__

**Source:** `packages/tools/src/tools/ml_models/__main__.py`
**Kind:** module

## Purpose

This module is the `python -m tools.ml_models` entry shim.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `main` | function | Imported from `tools.ml_models.cli` |

## Inputs and outputs

When executed, the module calls `main()` and exits with its result.

## Behavior

1. Import `main` from `tools.ml_models.cli`.
2. Call `main()` when the module runs as `__main__`.

## Errors and faults

Exit codes and errors come from `tools.ml_models.cli`.

## Messages

None.

## Configuration

None.

## Constraints

This module contains no command logic.

## Related documents

- [`tools.ml_models`](../ml_models.md)
- [`tools.ml_models.cli`](cli.md)
