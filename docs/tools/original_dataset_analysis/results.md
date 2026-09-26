# tools.original_dataset_analysis.results

**Source:** `packages/tools/src/tools/original_dataset_analysis/results.py`
**Kind:** module

## Purpose

This module re-exports table writers from `tools.ml_models.analysis.results`.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `write_stub_tables` | function | Markdown tables with blank scores |
| `write_filled_tables` | function | Native scores filled |

## Inputs and outputs

`write_stub_tables(order, path) -> None`.

`write_filled_tables(order, scores, path) -> None`.

## Behavior

1. The native section has one row per native cell.
2. The ground-sample section has one row per 12-band and RGB cell at each legal side.
3. ``write_stub_tables`` leaves every score empty. ``write_filled_tables`` writes
   a score when the key is present, including a 120 px score on the matching
   ground-sample row.

## Errors and faults

None from this module. The parent directory is created when it is absent.

## Messages

None.

## Configuration

None.

## Constraints

The function does not train a model and does not read an archive.

## Related documents

- [`tools.original_dataset_analysis`](../original_dataset_analysis.md)
- [`tools.original_dataset_analysis.matrix`](matrix.md)
