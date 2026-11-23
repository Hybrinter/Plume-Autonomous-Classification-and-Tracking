# tools.ml_models.studies

**Source:** `packages/tools/src/tools/ml_models/studies/`
**Kind:** package

## Purpose

The studies package holds offline design studies over the shared dataset
sources. Studies reuse the canonical Zenodo archive, band, resample, and
split helpers; they do not build finished datasets.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`band_matrix`](studies/band_matrix.md) | package | Zenodo 4250706 native-band and GSD sweep |

## Package interface

`tools.ml_models.studies.__init__` carries a module docstring only.
Callers import the study packages directly; the band-matrix study runs
through `python -m tools.ml_models.studies.band_matrix` or
`scripts/gsd_sweep.py`.

## Interactions

Studies read the Zenodo tar archives through
`tools.ml_models.dataset.sources.zenodo.archive` and apply the shared
`SplitRecipe`/`assign_group_splits` recipe from
`tools.ml_models.dataset.split`. The package does not import `flight`
or `tools.analysis`.

## Constraints

- Studies are read-only over the corpus archives; caches write to local
  directories chosen by the caller.
- No finished-dataset construction happens inside studies.

## Related documents

- [`tools.ml_models`](../ml_models.md)
- [`tools.ml_models.dataset`](../dataset.md)
- [`tools`](../tools.md)
