# tools.analysis.plots.common

**Source:** `packages/tools/src/tools/analysis/plots/common.py`
**Kind:** module

## Purpose

Common supplies shared matplotlib primitives for per-group figure builders. Rendering uses the
headless Agg backend and deterministic wide frames.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `LabeledFigure` | class | Named, titled matplotlib `Figure` |
| `present_numeric` | function | Columns with at least one finite value |
| `line_panel` | function | Multi-series line plot versus step |
| `categorical_timeline` | function | Step timeline with ordinal label mapping |
| `stacked_counts` | function | Stacked area of per-step count columns |
| `cumulative_lines` | function | Line panel of `.cumulative` derivations |
| `value_with_limit` | function | Value series with dashed limit overlay |
| `save_figures` | function | Write PNGs and close figures |

## Inputs and outputs

Each constructor accepts a group's wide DataFrame (step-indexed, leading `t` column) and
returns `LabeledFigure | None` when no renderable data exists.

**`save_figures(figures, out_dir) -> list[Path]`**

- Output: written PNG paths in input order.

## Behavior

1. `matplotlib.use("Agg")` runs at import time.
2. `line_panel` plots finite numeric columns with optional legend.
3. `categorical_timeline` maps distinct labels to ordinal y values with a step plot.
4. `stacked_counts` skips columns whose sum is zero.
5. `value_with_limit` overlays a dashed limit line when present.
6. `save_figures` writes `<name>.png` per figure and closes it.

## Errors and faults

None.

## Messages

None.

## Configuration

Figure size is `(9.0, 4.5)` at 110 DPI.

## Constraints

- Legend labels strip the leading `group.` prefix from signal names.
- Empty or all-NaN columns produce no figure (caller filters `None`).

## Related documents

- [`tools.analysis.plots`](plots.md)
- [`tools.analysis.report`](report.md)
