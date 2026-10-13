# tools.ml_models.analysis.generalization_figures

**Source:** `packages/tools/src/tools/ml_models/analysis/generalization_figures.py`
**Kind:** module

## Purpose

Stratum, interval, baseline, and heatmap chart recipes copied from
frozen generalization evidence. No resampling, source loading,
model evaluation, baseline fitting, or metric reduction occurs.
Missing-metadata cohorts keep explicit labels and support; grouped
random holdouts are not presented as temporal or unseen-sensor
proof.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `generalization_figure_data` | function | Copy frozen generalization evidence into chart recipes |

## Inputs and outputs

`generalization_figure_data(evidence, frozen) ->
Result[tuple[ModelFigure, ...], str]` requires the recorded split
identity of the frozen `GeneralizationEvidence` snapshot to match
the supplied `SplitEvidence` exactly.

## Behavior

Emitted figures include one `grouped_metric_*` figure per frozen
metric with copied interval endpoints and warnings; one stratum
figure per stratum/metric pair with copied estimates, confidence
intervals, support annotations, and explicit
`(recorded metadata missing)` cohorts, paginated to at most six
categories per page with every category retained; `heatmap_*`
figures over recorded GSD/truth-size strata tiled into pages of at
most 6x6 cells that all share the frozen full-matrix
`matrix_range`; `baseline_*` point comparisons against the
training-derived baseline with per-point support; and frozen
`baseline_curve_*` charts that retain the copied curve
unavailability reason, including all-null captures. Absent cells
and null estimates stay null; no value is resampled or
re-bootstrapped.

## Errors and faults

Returns `Err` for split-identity mismatches, strata carrying mixed
metric definitions, GSD-size strata without exactly two recorded
category axes, or repeated heatmap cells.

## Messages

None.

## Configuration

None.

## Constraints

- All pages preserve every recorded category; pagination never
  drops strata.
- Confidence intervals and selected operating points are copied
  verbatim; heatmap color bounds come from the frozen full matrix.

## Related documents

- [`tools.ml_models.analysis`](../analysis.md)
- [`tools.ml_models.analysis.model_figures`](model_figures.md)
- [`tools.ml_models.analysis.generalization_artifacts`](generalization_artifacts.md)
- [`tools.ml_models.analysis.plots.generalization`](plots/generalization.md)
