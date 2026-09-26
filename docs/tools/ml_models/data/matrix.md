# tools.ml_models.data.matrix

**Source:** `packages/tools/src/tools/ml_models/data/matrix.py`
**Kind:** module

## Purpose

This module lists the native-resolution runs and the ground-sample runs, and
refuses a result table that omits a planned cell.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `Cell` | class | Task, subset name, and side |
| `native_cells` | function | The native matrix |
| `gsd_cells` | function | 12-band set and RGB at every legal side |
| `require_complete` | function | Completeness check |

## Inputs and outputs

`native_cells(order) -> tuple[Cell, ...]`.

`gsd_cells(order) -> tuple[Cell, ...]`.

`require_complete(rows, expected) -> None`.

## Behavior

1. Subsets are the 12-band set, RGB, and one leave-one-out cell per 12-band
   id. B10 is not a cell.
2. Each subset is listed for the classifier and the segmentor at side 120.
   That is 28 cells.
3. A missing cell raises. The message includes its task, subset, and side.
4. `gsd_cells` repeats the 12-band set and RGB at sides 120, 80, 60, 40, and
   30 for both tasks. Those sides are 10, 15, 20, 30, and 40 metres.

## Errors and faults

`ValueError` from `require_complete` when a planned cell is absent.

## Messages

None.

## Configuration

None.

## Constraints

Leave-one-out names come from the verified 12-band set. B10 is not a dropout.
Side 76 is not a matrix side.

## Related documents

- [`tools.ml_models.data`](../data.md)
- [`tools.ml_models.data.bands`](bands.md)
- [`tools.ml_models.data.grid`](grid.md)
