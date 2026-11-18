# tools.ml_models.fusion.__main__

**Source:** `packages/tools/src/tools/ml_models/fusion/__main__.py`
**Kind:** module

## Purpose

This module starts the command line when the package is run as a program.

## Public interface

None.

## Inputs and outputs

None.

## Behavior

1. `python -m tools.ml_models.fusion` calls `cli.main`.
2. The process exit code is that function's return value.

## Errors and faults

Exceptions from `cli.main` propagate.

## Messages

None.

## Configuration

None.

## Constraints

The module only delegates to the command line.

## Related documents

- [`tools.ml_models.fusion`](../fusion.md)
- [`tools.ml_models.fusion.cli`](cli.md)
