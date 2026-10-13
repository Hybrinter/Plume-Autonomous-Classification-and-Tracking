# tools.ml_models.analysis.plots.common

**Source:** `packages/tools/src/tools/ml_models/analysis/plots/common.py`
**Kind:** module

## Purpose

This module holds figure/export conventions and the render-only
boundary for frozen evidence directories. `export_figure` encodes one
rendered figure into typed bundle bytes; `render_analysis` re-renders
a verified published bundle from its frozen recipes into a fresh
exclusive output.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `export_figure` | function | Deterministic per-format figure encoding to bundle bytes |
| `render_analysis` | function | Render-only boundary over a verified frozen evidence bundle |

## Inputs and outputs

`export_figure(figure: matplotlib.figure.Figure, identifier: str, cfg:
PlotConfig, *, kind: ArtifactKind = "FIGURE", population: str | None =
None) -> Result[tuple[BundleFile, ...], str]` returns one `BundleFile`
per configured format. `FIGURE` files use `figures/<identifier>.<fmt>`;
`VISUAL` files use `visuals/<identifier>.<fmt>`.

`render_analysis(evidence_dir: Path, cfg: PlotConfig, out: Path) ->
Result[Path, str]` checksum-verifies the bundle, then dispatches on
the summary kind: `ModelTrainingSummary` bundles go to
`model_render.render_model_bundle` and `DatasetSummary` bundles to
`dataset_render.render_dataset_bundle`. The `render` CLI command calls
it.

## Behavior

- `export_figure` saves through in-memory `BytesIO` buffers only; it
  never reserves or writes a user output path. Each path is validated
  by `BundleFile`.
- The supplied figure is closed in a `finally` block, including after a
  failed `savefig`, so no pyplot handle leaks.
- Matplotlib imports lazily and selects the headless `Agg` backend.
  A fixed SVG hash salt and omitted date metadata keep identical
  content byte-stable where the backend permits.
- `render_analysis` reads only referenced files and versioned frozen
  documents; a missing or corrupt recipe returns an actionable
  fresh-analyze-required error, and no output is created before
  publication. A failure during publication retains the reserved
  directory with an `.incomplete` marker. The destination is exclusive
  and never inside the bundle.

## Errors and faults

`export_figure` returns `Err` when a `savefig` call raises an
`OSError`, `ValueError`, or `RuntimeError`. `render_analysis` returns
`Err` when bundle verification fails, the summary kind is unsupported,
or the dispatched render boundary fails.

## Messages

None.

## Configuration

`PlotConfig` (`formats`, `dpi`, `width_inches`, `height_inches`,
`font_size`); see
[`tools.ml_models.analysis.config`](../config.md).

## Constraints

- Rendering consumes frozen evidence only and never reruns inference.
- No figure directory is created before the bundle verifies and every
  renderer has succeeded.

## Related documents

- [`tools.ml_models.analysis.plots`](../plots.md)
- [`tools.ml_models.cli`](../cli.md)
