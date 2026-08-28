# tools.inference.plots

**Source:** `packages/tools/src/tools/inference/plots.py`
**Kind:** module

## Purpose

This module builds headless matplotlib figures from a run directory. It does not
import `tools.analysis`.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `LabeledFigure` | class | Named Figure ready to save |
| `history_figures` | function | Train/val curves from `history.csv` |
| `overlay_figures` | function | Input / gold / prediction panels |
| `failure_figures` | function | Lowest-scoring preview samples |
| `save_figures` | function | PNG emission into a directory |

## Inputs and outputs

`history_figures(history_csv) -> list[LabeledFigure]`.

`overlay_figures(predictions_npz, limit=4) -> list[LabeledFigure]`.

`save_figures(figures, out_dir) -> list[Path]`.

## Behavior

1. Select the Agg backend on import.
2. Draw one curve figure per numeric history column.
3. Draw overlay panels from `predictions.npz` when that file exists.
4. Draw a failure gallery when any preview score is below 1.0.

## Errors and faults

`ValueError` when an overlay image is not rank 3.

## Messages

None.

## Configuration

Figure size is 9.0 by 4.5 inches at 110 DPI for curves.

## Constraints

No display is required. This module does not import `tools.analysis`. Overlay
figures read numpy arrays from `predictions.npz`.

## Related documents

- [`tools.inference.report`](report.md)
- [`tools.inference.eval`](eval.md)
