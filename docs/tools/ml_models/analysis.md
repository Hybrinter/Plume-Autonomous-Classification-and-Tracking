# tools.ml_models.analysis

**Source:** `packages/tools/src/tools/ml_models/analysis/`
**Kind:** package

## Purpose

The analysis package holds the evidence-first analysis scaffold: typed
evidence records, analysis and evaluation config, the bounded capture
sink, the exhaustive split evaluation, remaining unavailable entry
boundaries, pure metric/plot/visual scaffolds, and the retained pure
helpers (parameter counting, frontier and knee selection, row
formatters).

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`contracts`](analysis/contracts.md) | module | Strict sample-key, metric, curve, and identity records |
| [`config`](analysis/config.md) | module | Strict config records, TOML codecs, and digests |
| [`artifacts`](analysis/artifacts.md) | module | Versioned codecs, identities, publication, and typed tables |
| [`capture`](analysis/capture.md) | module | Bounded prediction/evidence sink |
| [`evaluate`](analysis/evaluate.md) | module | Exhaustive split-evaluation boundary |
| [`dataset`](analysis/dataset.md) | module | Mask/pixel measurement, whole-dataset measurement, and summary assembly |
| [`dataset_artifacts`](analysis/dataset_artifacts.md) | module | Frozen-measurement bundle persistence and publication |
| [`dataset_figures`](analysis/dataset_figures.md) | module | Frozen dataset figure-coordinate recipes |
| [`dataset_previews`](analysis/dataset_previews.md) | module | Bounded exact preview capture with semantic channel mapping |
| [`dataset_render`](analysis/dataset_render.md) | module | Figure/preview orchestration and rendered-bundle publication |
| [`training`](analysis/training.md) | module | Strict training records, epoch reduction, selection, and history reader |
| [`model`](analysis/model.md) | module | Model-analysis boundary (unavailable) |
| [`summaries`](analysis/summaries.md) | module | Tagged versioned summary records |
| [`cost`](analysis/cost.md) | module | Parameter counts and a partial two-input operation bound |
| [`runs`](analysis/runs.md) | module | Unavailable catalog readers; pure formatters |
| [`pareto`](analysis/pareto.md) | module | Pure frontier/knee helpers; unavailable reader boundary |
| [`metrics`](analysis/metrics.md) | package | Pure metric cores; classifier/calibration/segmentation/spatial implemented |
| [`plots`](analysis/plots.md) | package | Figure export and dataset figure renderers; the general render boundary stays unavailable |
| [`visuals`](analysis/visuals.md) | package | Gallery selection and dataset preview-gallery rendering |

## Package interface

`tools.ml_models.analysis.__init__` carries a module docstring only.
Callers import the leaf modules.

## Interactions

`evaluate` implements exhaustive split scoring. `dataset` measures a
finished dataset into a frozen `DatasetMeasurement`, then
`dataset_render` captures exact previews, derives frozen chart recipes
once, renders them through `plots.dataset` and `visuals.dataset`, and
publishes one checksummed evidence bundle through
`dataset_artifacts`. The `dataset analyze` CLI command calls that path
directly. The `model` boundary and the `plots.common` render boundary
remain unavailable; `render` fails closed.
`cost` profiles torch modules from `tools.ml_models.arch.registry`.
`runs` and `pareto` format helpers consume caller-supplied rows.

## Constraints

- Unavailable boundaries return explicit `Err` values and create no
  outputs; no dummy numbers or empty-success results are produced.
- Torch imports happen inside the modules that need them.

## Related documents

- [`tools.ml_models`](../ml_models.md)
- [`tools.ml_models.dataset`](../dataset.md)
- [`tools`](../tools.md)
