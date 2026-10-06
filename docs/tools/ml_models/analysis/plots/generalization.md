# tools.ml_models.analysis.plots.generalization

**Source:** `packages/tools/src/tools/ml_models/analysis/plots/generalization.py`
**Kind:** module

## Purpose

Generalization-evidence figure export over the shared model
renderer. A thin family binding: every supplied recipe is drawn
verbatim by `plots.model` and namespaced under
`figures/generalization/<split>/`.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `render_generalization_figures` | function | Export frozen generalization recipes |

## Inputs and outputs

`render_generalization_figures(figures, cfg) ->
Result[RenderedDatasetFigures, str]` calls
`render_model_figures(figures, cfg, family="generalization")`.
Files land under
`figures/generalization/<split>/<identifier>.<fmt>`; outputs are
named `model_figure:generalization:<split>:<identifier>`.

## Behavior

All drawing, validation, and availability semantics come from
`plots.model`; this module only fixes the `generalization` family
namespace.

## Errors and faults

Every `Err` from the shared renderer propagates unchanged.

## Messages

None.

## Configuration

`PlotConfig` controls dimensions, DPI, font size, and formats.

## Constraints

- No generalization-specific drawing lives here; the shared
  renderer owns all semantics.

## Related documents

- [`tools.ml_models.analysis.plots`](../plots.md)
- [`tools.ml_models.analysis.plots.model`](model.md)
- [`tools.ml_models.analysis.generalization_figures`](../generalization_figures.md)
