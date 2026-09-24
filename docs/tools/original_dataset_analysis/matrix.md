# tools.original_dataset_analysis.matrix

**Source:** `packages/tools/src/tools/original_dataset_analysis/matrix.py`
**Kind:** module

## Purpose

This module lists the native-resolution runs and refuses a result table that
omits one of them.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `Cell` | class | Task, subset name, and side |
| `native_cells` | function | The native matrix |
| `require_complete` | function | Completeness check |

## Inputs and outputs

`native_cells(order) -> tuple[Cell, ...]`.

`require_complete(rows, expected) -> None`.

## Behavior

1. Subsets are ceiling, RGB, one leave-one-out cell per ceiling band, and the
   13-band set.
2. Each subset is listed for the classifier and the segmentor at side 120.
3. A missing cell raises. The message includes its task, subset, and side.

## Errors and faults

`ValueError` from `require_complete` when a planned cell is absent.

## Messages

None.

## Configuration

None.

## Constraints

Leave-one-out names come from the verified ceiling. B10 is not a dropout.

## Related documents

- [`tools.original_dataset_analysis`](../original_dataset_analysis.md)
- [`tools.original_dataset_analysis.cli`](cli.md)
