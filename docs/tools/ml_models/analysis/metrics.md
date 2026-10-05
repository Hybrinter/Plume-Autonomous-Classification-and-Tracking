# tools.ml_models.analysis.metrics

**Source:** `packages/tools/src/tools/ml_models/analysis/metrics/`
**Kind:** package
**Status:** stub

## Purpose

The metrics package is the scaffold for deterministic pure metric cores
and their declarative definitions. Every submodule is unimplemented and
no metric algorithm executes yet.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`definitions`](metrics/definitions.md) | module | Metric names, validity, aggregation, and unit metadata (scaffold) |
| [`classifier`](metrics/classifier.md) | module | Binary counts, ranking, and operating curves (scaffold) |
| [`segmentation`](metrics/segmentation.md) | module | Overlap, pixel, area, and loss records (scaffold) |
| [`calibration`](metrics/calibration.md) | module | Brier, reliability, and ECE diagnostics (scaffold) |
| [`localization`](metrics/localization.md) | module | Component matching and localization records (scaffold) |
| [`boundary`](metrics/boundary.md) | module | Boundary distances and tolerance scores (scaffold) |
| [`generalization`](metrics/generalization.md) | module | Strata and group-level uncertainty (scaffold) |

## Package interface

`tools.ml_models.analysis.metrics.__init__` carries a module docstring
only.

## Interactions

None yet.

## Constraints

- Metric cores will be pure functions over passed values; no I/O or
  inference lives here.

## Related documents

- [`tools.ml_models.analysis`](../analysis.md)
