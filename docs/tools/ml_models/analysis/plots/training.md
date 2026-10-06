# tools.ml_models.analysis.plots.training

**Source:** `packages/tools/src/tools/ml_models/analysis/plots/training.py`
**Kind:** module

## Purpose

This module renders frozen `TrainingFigure` recipes into bundle bytes.
It draws exactly the coordinates, markers, exposure counts, and labels
the recipes carry; no loss re-reduction, checkpoint re-selection,
interpolation, history parse, or model call happens here.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `training_figure` | function | Build one figure for artist inspection before export |
| `render_training_figures` | function | Export every recipe into bundle bytes and outputs |

## Inputs and outputs

`training_figure(record: TrainingFigure, cfg: PlotConfig) ->
Result[Figure, str]` returns one open headless figure.
`render_training_figures(figures: tuple[TrainingFigure, ...], cfg:
PlotConfig) -> Result[RenderedDatasetFigures, str]` exports every
recipe through `export_figure` under `figures/training/{identifier}`
and returns the files plus one availability record per recipe named
`training_figure:<identifier>`.

## Behavior

- Series draw their raw x/y coordinates verbatim with `marker="o"` so
  one-point series stay visible; duplicate x positions are preserved
  and nothing is smoothed or interpolated. `points_only` series render
  as scatter markers only — a selected-checkpoint test point can never
  become a line.
- Canonical train draws `#0072B2`, validation `#E69F00`, the selected
  test point `#009E73`; optimization and telemetry series draw dark
  gray `#555555`.
- Logarithmic axes mask source-null and nonpositive coordinates as NaN
  at the same array indices so masked positions break the line; no
  epsilon floor, clipping, or bridging occurs. Legend entries count
  unavailable (null) points separately from points omitted by the log
  domain. Image counts are labelled `IMAGE exposures=N`, other units
  `<UNIT> events=N`, and a legend title notes that counts span captured
  events, not unique images.
- Recorded markers draw as dashed vertical lines at their exact x
  positions; checkpoint labels show only the first eight hash
  characters. A marker outside the log x domain is omitted and named
  in the footer.
- Run status, the distinct recorded populations, and recipe warnings
  render as a wrapped footer on every figure, including placeholders;
  an incomplete run is never labelled completed.
- A recipe whose `reason` is set — or whose log-domain masking removes
  every finite point — renders a labelled placeholder naming that
  reason (`No captured points lie in the requested logarithmic domain`
  when a log axis excluded the captured points, `No captured points
  are available` otherwise) and is indexed `UNAVAILABLE`.

All figures use the configured dimensions and preserve the recorded
axis labels and scales.

## Errors and faults

Unsafe identifiers, unknown scales, x/y length mismatches, and
nonfinite coordinates or marker positions return `Err` before
drawing. Figure construction or `export_figure` failures abort the
batch so indexing always matches the files exactly; created figures
are closed on every path.

## Messages

None.

## Configuration

`PlotConfig`; see
[`tools.ml_models.analysis.config`](../config.md).

## Constraints

- Scientific values come from the frozen recipes only; restyling a
  figure cannot change a coordinate.
- Legend, footer, and annotation text describe recorded status and
  availability, never scientific conclusions or rankings.
- Exposure counts describe captured event populations; they are never
  presented as independent observation support.

## Related documents

- [`tools.ml_models.analysis.plots`](../plots.md)
- [`tools.ml_models.analysis.training_figures`](../training_figures.md)
- [`tools.ml_models.analysis.plots.common`](common.md)
