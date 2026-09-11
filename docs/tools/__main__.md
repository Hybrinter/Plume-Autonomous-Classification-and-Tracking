# tools.__main__

**Source:** `packages/tools/src/tools/__main__.py`
**Kind:** module

## Purpose

This module is the `python -m tools` entry shim.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `main` | function | Imported from `tools.cli` |

## Inputs and outputs

When executed, the module calls `main()` and exits with its result.

## Behavior

1. Import `main` from `tools.cli`.
2. Call `main()` when the module runs as `__main__`.

## Errors and faults

Exit codes and errors come from `tools.cli`.

## Messages

None.

## Configuration

None.

## Constraints

This module contains no command logic.

## Related documents

- [`tools`](../tools.md)
- [`tools.cli`](cli.md)
