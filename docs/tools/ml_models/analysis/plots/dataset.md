# tools.ml_models.analysis.plots.dataset

**Source:** `packages/tools/src/tools/ml_models/analysis/plots/dataset.py`
**Kind:** module

## Purpose

This module renders frozen `DatasetFigure` recipes into bundle bytes.
It draws exactly the coordinates, support counts, and labels the
recipes carry; no histogram binning, cumulative computation, rate
derivation, source read, or model call happens here.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `RenderedDatasetFigures` | class | Rendered bundle files plus per-figure availability records |
| `dataset_figure` | function | Build one figure for artist inspection before export |
| `render_dataset_figures` | function | Export every recipe into bundle bytes and outputs |

## Inputs and outputs

`dataset_figure(record: DatasetFigure, cfg: PlotConfig) -> Result[Figure,
str]` returns one open headless figure. `render_dataset_figures(figures:
tuple[DatasetFigure, ...], cfg: PlotConfig) ->
Result[RenderedDatasetFigures, str]` exports every recipe through
`export_figure` and returns the files plus one availability record per
recipe named `figure:<identifier>`.

## Behavior

Figure kinds are handled by explicit conditional branches:

- `BAR`: the union of categorical x values across series is drawn in
  recipe order with aligned positions. Only non-null y values draw
  bars; a null value is annotated `n/a` at its category and never drawn
  as a zero bar. Series use distinct colorblind-safe colors (train
  `#0072B2`, val `#E69F00`, test `#009E73`); legend entries carry the
  support unit and `n`.
- `ECDF`: supplied x/y coordinates are drawn with `ax.step(...,
  where="post")` unchanged. An empty `n=0` series stays in the legend
  explicitly marked unavailable; no artificial curve is created.
- `HISTOGRAM`: supplied counts and edges are drawn with `ax.stairs`;
  no rebinning occurs.
- `MATRIX`: null cells are masked gray and labelled `n/a`; captured
  values display on a fixed Pearson `-1..1` scale in the original band
  label order with the matrix support shown on the colorbar.
- A recipe carrying `reason` renders a standalone placeholder naming
  the title and reason with no fabricated data; its bytes are still
  indexed and its output record is `UNAVAILABLE` with that reason.

All figures use the configured dimensions (no tight-bbox cropping) and
preserve the labelled population and axis units from the recipe.

## Errors and faults

Returns `Err` when figure construction or `export_figure` fails; a
failed recipe aborts the batch so summary indexing always matches the
files exactly.

## Messages

None.

## Configuration

`PlotConfig`; see
[`tools.ml_models.analysis.config`](../config.md).

## Constraints

- Scientific values come from the frozen recipes only; restyling a
  figure cannot change a measurement.
- Legend and annotation text describe support and availability, never
  scientific conclusions.

## Related documents

- [`tools.ml_models.analysis.plots`](../plots.md)
- [`tools.ml_models.analysis.dataset_figures`](../dataset_figures.md)
- [`tools.ml_models.analysis.plots.common`](common.md)
