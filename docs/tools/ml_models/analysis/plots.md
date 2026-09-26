# tools.ml_models.analysis.plots

**Source:** `packages/tools/src/tools/ml_models/analysis/plots.py`
**Kind:** module

## Purpose

This module writes headless matplotlib PNGs for training runs, chip previews,
classifier scores, full-frame metrics, blobs, and the band study.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `LabeledFigure` | class | Named Figure ready to save |
| `history_figures` | function | Train and val curves from `history.csv` |
| `overlay_figures` | function | Input, gold, and prediction panels |
| `failure_figures` | function | Lowest-scoring preview samples |
| `save_figures` | function | PNG emission into a directory |
| `write_band_bars` | function | One bar per subset name |
| `write_metric_bars` | function | One chart for one metric |
| `write_delta_bars` | function | Score minus the 12-band baseline |
| `write_pr_curve` | function | Precision-recall curve |
| `write_loss_curve` | function | Train, validation, and test loss |
| `write_learning_rate` | function | Learning rate versus epoch |
| `write_gsd_lines` | function | `pr_auc` and `native_dice` versus distance |
| `write_canvas_preview` | function | One canvas image and its mask |
| `write_score_histogram` | function | Histogram of classifier scores |
| `write_reliability` | function | Binned probability versus outcome |
| `write_hit_rate_by_placement` | function | Center, corner, and edge hit-rate bars |
| `write_empty_fpr` | function | Empty-frame false-positive rate |
| `write_logit_margin` | function | Chip max logit versus frame max logit |
| `write_blob_area_histogram` | function | Histogram of blob area |

## Inputs and outputs

`history_figures(history_csv) -> list[LabeledFigure]`.

`save_figures(figures, out_dir) -> list[Path]`.

Each `write_*` function saves one PNG and returns that path.

`write_gsd_lines(scores, path) -> Path`. The classify axis is labeled
`pr_auc`. The segment axis is labeled `native_dice`.

## Behavior

1. Select the Agg backend on import.
2. `history_figures` draws one curve per numeric history column, including
   loss and a validation metric when those columns are present.
3. `write_loss_curve` draws train, validation, and test loss on a log axis.
4. `write_learning_rate` draws learning rate against epoch.
5. `write_canvas_preview` draws the image and the mask with the mask scale
   fixed from 0 to 1, so a soft border stays visible.
6. `write_reliability` bins probabilities and plots mean outcome.
7. `write_gsd_lines` draws two panels. Coarse Dice is not on the `native_dice`
   axis.

## Errors and faults

`ValueError` when a series is empty or misaligned, when a loss is not
positive, or when a ground-sample series is missing a point.

## Messages

None.

## Configuration

Curve figures use 9.0 by 4.5 inches at 110 DPI.

## Constraints

No display is required. This module does not import `tools.analysis`.

## Related documents

- [`tools.ml_models.analysis`](../analysis.md)
- [`tools.ml_models.analysis.report`](report.md)
- [`tools.ml_models.analysis.full_frame`](full_frame.md)
