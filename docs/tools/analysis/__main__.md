# tools.analysis.__main__

**Source:** `packages/tools/src/tools/analysis/__main__.py`
**Kind:** module

## Purpose

The module entry dispatches `python -m tools.analysis` to the CLI `main` function.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `main` | function | Imported from `tools.analysis.cli`; invoked when run as `-m` |

## Inputs and outputs

When executed as `__main__`, calls `sys.exit(main())` with process argv.

## Behavior

1. Import `main` from `tools.analysis.cli`.
2. Call `sys.exit(main())` when `__name__ == "__main__"`.

## Errors and faults

Exit code comes from the CLI dispatcher (0 on success).

## Messages

None.

## Configuration

None.

## Constraints

- Thin shim only. All command logic lives in `tools.analysis.cli`.

## Related documents

- [`tools.analysis`](analysis.md)
- [`tools.analysis.cli`](cli.md)
