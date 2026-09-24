# tools.original_dataset_analysis.plots

**Source:** `packages/tools/src/tools/original_dataset_analysis/plots.py`
**Kind:** module

## Purpose

This module writes a horizontal bar chart of one score per subset.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `write_band_bars` | function | PNG bar chart |

## Inputs and outputs

`write_band_bars(names, scores, path) -> None`.

## Behavior

1. One bar is drawn per name. The horizontal axis runs from 0 to 1.
2. The parent directory of ``path`` is created when it is absent.
3. The figure is closed after it is saved.

## Errors and faults

`ValueError` when ``names`` and ``scores`` differ in length or are empty.

## Messages

None.

## Configuration

None.

## Constraints

The chart is written with the Agg backend.

## Related documents

- [`tools.original_dataset_analysis`](../original_dataset_analysis.md)
- [`tools.original_dataset_analysis.matrix`](matrix.md)
