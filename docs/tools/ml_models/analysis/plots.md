# tools.ml_models.analysis.plots

**Source:** `packages/tools/src/tools/ml_models/analysis/plots.py`
**Kind:** module

## Purpose

This module renders training-run figures: loss and metric history from
`history.csv`, plus prediction overlays and failure mosaics from
`predictions.npz`.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `LabeledFigure` | dataclass | One figure with its output stem |
| `save_figures` | function | Write figures to a directory as PNG |
| `history_figures` | function | Loss and metric curves from `history.csv` |
| `overlay_figures` | function | Prediction overlays from `predictions.npz` |
| `failure_figures` | function | Worst-case failure mosaics |

## Inputs and outputs

Figure builders take run artifact paths and return `LabeledFigure` lists;
`save_figures` writes PNG files and returns their paths.

## Behavior

Matplotlib uses the non-interactive `Agg` backend. `history_figures`
reads the CSV; `overlay_figures` and `failure_figures` read the
probability arrays and apply `sigmoid` from `train.metrics` to logits.

## Errors and faults

Missing artifacts raise `FileNotFoundError`; malformed arrays surface
as numpy errors.

## Messages

None.

## Configuration

None.

## Constraints

- The module imports matplotlib eagerly with the `Agg` backend.
- Figures close after saving; no GUI is required.

## Related documents

- [`tools.ml_models.analysis`](../analysis.md)
- [`tools.ml_models.analysis.report`](report.md)
- [`tools.ml_models.train.metrics`](../train/metrics.md)
