# tools.original_dataset_analysis.plots

**Source:** `packages/tools/src/tools/original_dataset_analysis/plots.py`
**Kind:** module

## Purpose

This module writes a horizontal bar chart of one score per subset.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `write_band_bars` | function | PNG bar chart |
| `write_metric_bars` | function | One chart for one metric |
| `write_delta_bars` | function | Difference from the 12-band score |
| `write_loss_curve` | function | Train, validation, and test loss |
| `write_pr_curve` | function | Precision-recall curve |

## Inputs and outputs

`write_band_bars(names, scores, path) -> None`.

`write_metric_bars(title, names, scores, path) -> None`.

`write_loss_curve(epochs, train_loss, val_loss, test_loss, path, selected_epoch) -> None`.

## Behavior

1. One bar is drawn per name. The horizontal axis runs from 0 to 1.
2. The parent directory of ``path`` is created when it is absent.
3. The figure is closed after it is saved.
4. ``write_delta_bars`` sorts the difference from the 12-band score and draws
   a zero line.
5. ``write_loss_curve`` uses a linear epoch axis and a log loss axis. Epochs
   are one-based. The vertical line is labeled with that epoch.
6. ``write_pr_curve`` draws recall on the horizontal axis and precision on the
   vertical axis.

## Errors and faults

`ValueError` when ``names`` and ``scores`` differ in length or are empty, or
when a loss series is empty, misaligned, or not strictly positive.

## Messages

None.

## Configuration

None.

## Constraints

The chart is written with the Agg backend.

## Related documents

- [`tools.original_dataset_analysis`](../original_dataset_analysis.md)
- [`tools.original_dataset_analysis.matrix`](matrix.md)
