# tools.cli

**Source:** `packages/tools/src/tools/cli.py`
**Kind:** module

## Purpose

The root CLI exposes installed engineering workflows through the `pact-tools`
command.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `app` | Typer application | Root tools command group |
| `main` | function | Console-script entry point |

## Inputs and outputs

`main(argv=None) -> int` accepts an optional argument vector and returns a
process exit code.

## Behavior

1. Register the inference application as `inference`.
2. Register the analysis application as `analysis`.
3. Dispatch the selected package command.

## Errors and faults

Invalid command input prints Click usage information and returns its nonzero
exit code. Package command errors retain their package-defined behavior.

## Messages

None.

## Configuration

None.

## Constraints

- Package command registration is explicit.
- The CLI does not use a plugin registry.
- Command logic remains in package-owned applications and library functions.

## Related documents

- [`tools`](../tools.md)
- [`tools.inference.cli`](inference/cli.md)
- [`tools.analysis.cli`](analysis/cli.md)
