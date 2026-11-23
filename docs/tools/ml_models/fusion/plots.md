# tools.ml_models.fusion.plots

**Source:** `packages/tools/src/tools/ml_models/fusion/plots.py`
**Kind:** module

## Purpose

This module writes the figures of the VIS/SWIR fusion trade.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `write_seed_plot` | function | One dot per seed and the seed mean for each option |
| `write_threshold_plot` | function | Test F1 per look-angle bin for each threshold method |

## Inputs and outputs

`write_seed_plot(scores, metric_label, title, path) -> None`.

`write_threshold_plot(rows, title, path) -> None`.

## Behavior

1. `write_seed_plot` draws each option in mapping order. Dots are seeds and a
   bar marks the mean, with the mean value next to it.
2. `write_threshold_plot` draws one line per method: fixed 0.5, one global
   threshold, and one threshold per look angle. Each line has its own marker
   shape. The title gives the F1 over all tiles for the three methods.
3. Both functions create the parent directory and write a PNG.

## Errors and faults

`write_seed_plot` raises `ValueError` when an option has no score.
`write_threshold_plot` raises `ValueError` when there is no bin row.

## Messages

None.

## Configuration

None.

## Constraints

The module uses the matplotlib Agg backend.

## Related documents

- [`tools.ml_models.fusion`](../fusion.md)
- [`tools.ml_models.fusion.thresholds`](thresholds.md)
- [`tools.ml_models.fusion.cli`](cli.md)
