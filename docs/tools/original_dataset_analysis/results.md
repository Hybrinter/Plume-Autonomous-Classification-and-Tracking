# tools.original_dataset_analysis.results

**Source:** `packages/tools/src/tools/original_dataset_analysis/results.py`
**Kind:** module

## Purpose

This module writes the native and ground-sample result tables with a blank
score column.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `write_stub_tables` | function | Markdown tables |

## Inputs and outputs

`write_stub_tables(order, path) -> None`.

## Behavior

1. The native section has one row per native cell.
2. The ground-sample section has one row per ceiling and RGB cell at each legal side.
3. The score cell is empty.

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
