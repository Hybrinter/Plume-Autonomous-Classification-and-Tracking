# tools.ml_models.analysis.model_figures

**Source:** `packages/tools/src/tools/ml_models/analysis/model_figures.py`
**Kind:** module

## Purpose

Frozen model-chart recipes and exact captured-row display reductions.
Coordinates, support, identities, and missingness are retained
separately from presentation. Curve recipes copy captured coordinates
without scoring; ECDFs reduce scalar evidence once with complete tie
grouping. Rendering consumes these recipes and never calls this module
to remeasure a bundle.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `DrawStyle` | type alias | `LINE`/`PRE`/`POST`/`BAR`/`POINT` draw conventions |
| `ModelFigureKind` | type alias | `SERIES`/`MATRIX` figure kinds |
| `FigureIdentity` | dataclass | Frozen dataset/manifest/checkpoint/task/split identity |
| `ModelSeries` | dataclass | Exact points, support, optional intervals, draw style |
| `ModelPoint` | dataclass | One captured operating point |
| `ModelFigure` | dataclass | Standalone recipe with identity, coordinates, availability |
| `figure_identity` | function | Copy a bound `SplitEvidence` identity |
| `safe_token` | function | Stable file token from a display value |
| `captured_values` | function | Read scalar metric values from a row |
| `validate_figure_rows` | function | Require complete unique canonical capture |
| `ecdf_series` | function | Freeze a right-continuous ECDF with complete ties |
| `distribution_figure` | function | Reduce caller-selected scalar populations once |
| `curve_figure` | function | Copy a captured curve verbatim with notes |
| `metric_figure` | function | Copy one scalar plus absolute interval endpoints |
| `curve_by_name` | function | Look up a captured curve without recomputation |

## Inputs and outputs

Inputs are frozen `SplitEvidence` and complete `CaptureRow` tuples.
Outputs are `ModelFigure` recipes or `Err` when capture is incomplete,
conflicting, or identity-mismatched. No model, dataset, or source is
read.

## Behavior

`ModelFigure` carries `identifier`, `identity`, axis labels,
`population`, verbatim `series`/`points`, optional `MATRIX` payloads
(`x_categories`, `y_categories`, `matrix`, `matrix_support`,
`matrix_range`), optional display-only `x_range`/`y_range` bounds, a
`reason`, and `notes`. Null cells and points remain
missing. A captured curve whose y values are all null keeps its raw
x/null coordinates and carries `reason` "No eligible captured values
for this curve", rendering as an explicit placeholder. Intervals keep
absolute endpoints and need not contain the point estimate.
`safe_token` produces stable path tokens without altering displayed
values.

## Errors and faults

`validate_figure_rows` returns `Err` for incomplete cohort coverage,
identity mismatches, or duplicate canonical variants.

## Messages

None.

## Configuration

None.

## Constraints

- ECDF scalar reductions happen once when recipes are frozen; no
  metric rescoring, threshold fitting, or ranking is derived. Renderers
  consume the frozen recipes and never call these reducers.
- Interval endpoints are absolute recorded bounds.

## Related documents

- [`tools.ml_models.analysis`](../analysis.md)
- [`tools.ml_models.analysis.classifier_figures`](classifier_figures.md)
- [`tools.ml_models.analysis.plots.model`](plots/model.md)
