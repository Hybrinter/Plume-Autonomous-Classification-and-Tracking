# tools.ml_models.analysis.plots.model

**Source:** `packages/tools/src/tools/ml_models/analysis/plots/model.py`
**Kind:** module

## Purpose

Shared model-figure rendering over frozen `ModelFigure` recipes.
Draws exact captured coordinates with recorded draw conventions;
interval bars use the recipe's absolute lower/upper endpoints; matrix
figures mask null cells, annotate exact support counts, and use the
supplied `matrix_range` or the data bounds. No coordinate is
smoothed, filtered, interpolated, or recomputed.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `model_figure` | function | Render one recipe into a `Result[Figure, str]` |
| `render_model_figures` | function | Export recipes under `figures/<family>/<split>/` |

## Inputs and outputs

`render_model_figures(figures, cfg, *, family)` returns
`RenderedDatasetFigures` with `files` under
`figures/<family>/<split>/<identifier>.<fmt>` and outputs named
`model_figure:<family>:<split>:<identifier>`; `family` must be a
single safe relative token. Identities are unique per
`(split, identifier)` pair — the same identifier may render once in
each split.

## Behavior

`LINE`, `PRE` (steps-pre), and `POST` (steps-post) series draw as
lines with `o` markers so singletons stay visible; null y values
become NaN gaps that break the line without filtering duplicate x
positions. `POINT` renders markers only and `BAR` draws recorded
heights at the frozen x coordinates (never re-indexed); a null height
stays absent with an `n/a` note anchored at zero and category ticks
sit at the recorded x positions. Series `lower`/`upper` draw as
absolute vline endpoints with caps, never `yerr` (intervals need not
enclose the estimate). `point_support` annotates per-point counts.
Recorded baseline series (perfect calibration, chance or random
diagonals/lift, cohort prevalence) draw dashed and behind the curves.
Matrix figures mask null cells gray with `n/a`, annotate exact
support counts with contrast-adaptive text, use recorded
`matrix_range` or data bounds, and label truth/prediction axes from
the recipe. Operating `ModelPoint`s draw as exact markers with their
recorded names. Recipes with a recorded `reason` render explicit
placeholder panels and index `UNAVAILABLE`. Population and notes wrap
into a layout-managed footer on every figure kind.

## Errors and faults

Malformed shapes (mismatched lengths, misaligned intervals or
support, unordered or unpaired interval endpoints, bad matrix shapes
or ranges), non-finite coordinates, negative or boolean support
counts, unsafe identifiers/splits/families, and duplicate
`(split, identifier)` identities return `Err`; no silent zip
truncation occurs.

## Messages

None.

## Configuration

`PlotConfig` controls dimensions, DPI, font size, and formats.

## Constraints

- Coordinates are copied verbatim; rendering never recomputes,
  ranks, or resamples.
- Interval endpoints stay absolute recorded bounds.
- Availability is copied from the recipe reason.

## Related documents

- [`tools.ml_models.analysis.plots`](../plots.md)
- [`tools.ml_models.analysis.model_figures`](../model_figures.md)
- [`tools.ml_models.analysis.plots.classifier`](classifier.md)
