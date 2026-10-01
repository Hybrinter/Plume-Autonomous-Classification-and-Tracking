# tools.ml_models.studies.band_matrix.__main__

**Source:** `packages/tools/src/tools/ml_models/studies/band_matrix/__main__.py`
**Kind:** module

## Purpose

This module starts the command line when the package is run as a program.

## Public interface

None.

## Inputs and outputs

None.

## Behavior

1. ``python -m tools.ml_models.studies.band_matrix`` calls :func:`cli.main`.
2. The process exit code is that function's return value.

## Errors and faults

Exceptions from :func:`cli.main` propagate.

## Messages

None.

## Configuration

None.

## Constraints

The module only delegates to the command line.

## Related documents

- [`tools.ml_models.studies.band_matrix`](../band_matrix.md)
- [`tools.ml_models.studies.band_matrix.cli`](cli.md)
