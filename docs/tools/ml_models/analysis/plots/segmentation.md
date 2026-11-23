# tools.ml_models.analysis.plots.segmentation

**Source:** `packages/tools/src/tools/ml_models/analysis/plots/segmentation.py`
**Kind:** module

## Purpose

Segmentation-evidence figure export over the shared model renderer.
A thin family binding: every supplied recipe is drawn verbatim by
`plots.model` and namespaced under `figures/segmentor/<split>/` so
artifact paths align with the recorded `segmentor` task identity.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `render_segmentation_figures` | function | Export frozen segmentation recipes |

## Inputs and outputs

`render_segmentation_figures(figures, cfg) ->
Result[RenderedDatasetFigures, str]` calls
`render_model_figures(figures, cfg, family="segmentor")`. Files land
under `figures/segmentor/<split>/<identifier>.<fmt>`; outputs are
named `model_figure:segmentor:<split>:<identifier>`.

## Behavior

All drawing, validation, and availability semantics come from
`plots.model`; this module only fixes the `segmentor` family
namespace.

## Errors and faults

Every `Err` from the shared renderer propagates unchanged.

## Messages

None.

## Configuration

`PlotConfig` controls dimensions, DPI, font size, and formats.

## Constraints

- No segmentation-specific drawing lives here; the shared renderer
  owns all semantics.

## Related documents

- [`tools.ml_models.analysis.plots`](../plots.md)
- [`tools.ml_models.analysis.plots.model`](model.md)
- [`tools.ml_models.analysis.segmentation_figures`](../segmentation_figures.md)
