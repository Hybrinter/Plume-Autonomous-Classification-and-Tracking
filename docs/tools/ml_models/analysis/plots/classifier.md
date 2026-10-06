# tools.ml_models.analysis.plots.classifier

**Source:** `packages/tools/src/tools/ml_models/analysis/plots/classifier.py`
**Kind:** module

## Purpose

Classifier-evidence figure export over the shared model renderer. A
thin family binding: every supplied recipe is drawn verbatim by
`plots.model` and namespaced under `figures/classifier/<split>/`.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `render_classifier_figures` | function | Export frozen classifier recipes |

## Inputs and outputs

`render_classifier_figures(figures, cfg) ->
Result[RenderedDatasetFigures, str]` calls
`render_model_figures(figures, cfg, family="classifier")`. Files land
under `figures/classifier/<split>/<identifier>.<fmt>`; outputs are
named `model_figure:classifier:<split>:<identifier>`.

## Behavior

All drawing, validation, and availability semantics come from
`plots.model`; this module only fixes the `classifier` family
namespace.

## Errors and faults

Every `Err` from the shared renderer propagates unchanged.

## Messages

None.

## Configuration

`PlotConfig` controls dimensions, DPI, font size, and formats.

## Constraints

- No classifier-specific drawing lives here; the shared renderer owns
  all semantics.

## Related documents

- [`tools.ml_models.analysis.plots`](../plots.md)
- [`tools.ml_models.analysis.plots.model`](model.md)
- [`tools.ml_models.analysis.classifier_figures`](../classifier_figures.md)
