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
| [`generalization_artifacts`](analysis/generalization_artifacts.md) | module | Frozen generalization-evidence bundle codecs |
| [`training`](analysis/training.md) | module | Strict training records, epoch reduction, selection, and history reader |
| [`training_figures`](analysis/training_figures.md) | module | Frozen training-history figure-coordinate recipes |
| [`training_artifacts`](analysis/training_artifacts.md) | module | Frozen training-figure recipe bundle serialization |
| [`model_figures`](analysis/model_figures.md) | module | Frozen model-chart recipes and scalar reductions |
| [`classifier_figures`](analysis/classifier_figures.md) | module | Frozen classifier figure inventory |
| [`segmentation_figures`](analysis/segmentation_figures.md) | module | Frozen segmentor extent/localization figure inventory |
| [`generalization_figures`](analysis/generalization_figures.md) | module | Frozen stratum/baseline/heatmap figure copies |
| [`model_figure_artifacts`](analysis/model_figure_artifacts.md) | module | Frozen model-figure recipe bundle serialization |
| [`prediction_display`](analysis/prediction_display.md) | module | Verified segmentor display arrays from cached logits |
| [`prediction_selections`](analysis/prediction_selections.md) | module | Whole-cohort prediction gallery selection |
| [`prediction_artifacts`](analysis/prediction_artifacts.md) | module | Frozen prediction preview/manifest bundle codecs |
| [`model`](analysis/model.md) | module | Model-analysis boundary (unavailable) |
| [`summaries`](analysis/summaries.md) | module | Tagged versioned summary records |
| [`cost`](analysis/cost.md) | module | Parameter counts and a partial two-input operation bound |
| [`runs`](analysis/runs.md) | module | Unavailable catalog readers; pure formatters |
| [`pareto`](analysis/pareto.md) | module | Pure frontier/knee helpers; unavailable reader boundary |
| [`metrics`](analysis/metrics.md) | package | Pure metric cores; classifier/calibration/segmentation/spatial/generalization implemented |
| [`plots`](analysis/plots.md) | package | Figure export and dataset/training/model figure renderers; the general render boundary stays unavailable |
| [`visuals`](analysis/visuals.md) | package | Gallery selection and dataset/prediction preview-gallery rendering |

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
directly. `training_figures` freezes parsed run histories into chart
recipes once; `plots.training` renders those recipes into bundle bytes
and `training_artifacts` serializes the recipes themselves.
`classifier_figures` freezes split evidence and scalar capture into
model-chart recipes; `plots.model`/`plots.classifier` render them and
`model_figure_artifacts` serializes the recipes.
`segmentation_figures` freezes explicit-mask extent/localization
recipes and `plots.segmentation` renders them under the `segmentor`
family; `generalization_figures` copies frozen stratum evidence into
paginated recipes rendered by `plots.generalization`.
`prediction_selections` freezes whole-cohort gallery selections;
`visuals.predictions` renders them over verified preview bytes —
segmentor rows additionally pass through `prediction_display` for
frozen display arrays — and
`prediction_artifacts` serializes previews and the manifest. The `model`
boundary and the `plots.common` render boundary remain unavailable;
`render` fails closed.
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
