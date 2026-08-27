# tools.inference.__main__

**Source:** `packages/tools/src/tools/inference/__main__.py`
**Kind:** module

## Purpose

This module is the `python -m tools.inference` entry shim.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `main` | function | Imported from `tools.inference.cli` |

## Inputs and outputs

When executed, the module calls `main()` and exits with its result.

## Behavior

1. Import `main` from `tools.inference.cli`.
2. Call `main()` when the module runs as `__main__`.

## Errors and faults

Exit codes and errors come from `tools.inference.cli`.

## Messages

None.

## Configuration

None.

## Constraints

This module contains no command logic.

## Related documents

- [`tools.inference`](../inference.md)
- [`tools.inference.cli`](cli.md)
