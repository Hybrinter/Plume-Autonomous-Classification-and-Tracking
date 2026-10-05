# tools.ml_models.analysis.metrics

**Source:** `packages/tools/src/tools/ml_models/analysis/metrics/`
**Kind:** package

## Purpose

The metrics package holds deterministic pure metric cores and their
declarative definitions. The classifier, calibration, definitions, and
inputs modules are implemented; segmentation, localization, boundary,
and generalization remain scaffolds.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`definitions`](metrics/definitions.md) | module | Metric names, directions, formulas, and limitation metadata |
| [`inputs`](metrics/inputs.md) | module | Validated binary score/label vectors and stable transforms |
| [`classifier`](metrics/classifier.md) | module | Binary counts, ranking, and operating curves |
| [`calibration`](metrics/calibration.md) | module | Brier, reliability bins, and ECE diagnostics |
| [`segmentation`](metrics/segmentation.md) | module | Overlap, pixel, area, and loss records (scaffold) |
| [`localization`](metrics/localization.md) | module | Component matching and localization records (scaffold) |
| [`boundary`](metrics/boundary.md) | module | Boundary distances and tolerance scores (scaffold) |
| [`generalization`](metrics/generalization.md) | module | Strata and group-level uncertainty (scaffold) |

## Package interface

`tools.ml_models.analysis.metrics.__init__` carries a module docstring
only; callers import the concrete submodule APIs.

## Interactions

`classifier` composes `inputs` validation and `calibration` scoring;
`definitions` is standalone metadata. No orchestrator invokes these
cores yet.

## Constraints

- Metric cores are pure functions over passed values; no I/O or
  inference lives here.

## Related documents

- [`tools.ml_models.analysis`](../analysis.md)
