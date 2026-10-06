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
| [`training`](analysis/training.md) | module | Strict training records, epoch reduction, selection, and history reader |
| [`model`](analysis/model.md) | module | Model-analysis boundary (unavailable) |
| [`summaries`](analysis/summaries.md) | module | Tagged versioned summary records |
| [`cost`](analysis/cost.md) | module | Parameter counts and a partial two-input operation bound |
| [`runs`](analysis/runs.md) | module | Unavailable catalog readers; pure formatters |
| [`pareto`](analysis/pareto.md) | module | Pure frontier/knee helpers; unavailable reader boundary |
| [`metrics`](analysis/metrics.md) | package | Pure metric cores; classifier/calibration implemented |
| [`plots`](analysis/plots.md) | package | Figure-render scaffolds and the unavailable render boundary |
| [`visuals`](analysis/visuals.md) | package | Visual-evidence scaffolds |

## Package interface

`tools.ml_models.analysis.__init__` carries a module docstring only.
Callers import the leaf modules.

## Interactions

`evaluate` implements exhaustive split scoring. `dataset` measures a
finished dataset into a frozen `DatasetMeasurement` and publishes it as
a checksummed evidence bundle through `dataset_artifacts`; the `model`
and `plots.common` orchestration boundaries remain unavailable, and the
`dataset analyze` CLI command stays unavailable until rendering lands.
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
