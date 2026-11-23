# tools.ml_models.fusion

**Source:** `packages/tools/src/tools/ml_models/fusion/`
**Kind:** package

## Purpose

This package compares VIS/SWIR fusion methods on trained runs. It scores runs
on the test split, fuses separate VIS and SWIR models late, chooses classifier
thresholds per look angle, and draws the trade figures.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`thresholds`](fusion/thresholds.md) | module | Validation-chosen thresholds scored on test |
| [`late`](fusion/late.md) | module | Weighted logit sum of a VIS and a SWIR model |
| [`study`](fusion/study.md) | module | Split scoring and the late-fusion weight search |
| [`plots`](fusion/plots.md) | module | Seed and threshold figures |
| [`cli`](fusion/cli.md) | module | `test`, `late`, `thresholds`, and `seed-plot` commands |
| [`__main__`](fusion/__main__.md) | module | `python -m tools.ml_models.fusion` entry shim |

## Package interface

`tools.ml_models.fusion.__init__` carries a module docstring only. Callers
import the modules directly.

## Interactions

`study` calls `tools.ml_models.analysis.model_inputs` and
`tools.ml_models.analysis.evaluate`. `thresholds` reads the capture rows that
`python -m tools.ml_models analyze` writes. The package uses
`flight.libs.types` for the `Result` boundary.

## Constraints

- The package does not train models and does not write datasets.
- `test` and `late` run inference; do not run them during a training job.

## Related documents

- [`tools.ml_models`](../ml_models.md)
- [`tools.ml_models.analysis`](analysis.md)
