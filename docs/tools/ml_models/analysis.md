# tools.ml_models.analysis

**Source:** `packages/tools/src/tools/ml_models/analysis/`
**Kind:** package

## Purpose

The analysis package scores finished model artifacts and training runs:
parameter and FLOP accounting, run-directory summaries and ranking, Pareto
frontiers, and report rendering.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`cost`](analysis/cost.md) | module | Parameter and FLOP counters |
| [`runs`](analysis/runs.md) | module | Run discovery, summaries, and rank tables |
| [`pareto`](analysis/pareto.md) | module | Cost-versus-quality frontier and knee selection |
| [`plots`](analysis/plots.md) | module | History, overlay, and failure figures |
| [`report`](analysis/report.md) | module | `report.md` and figure bundle writer |

## Package interface

`tools.ml_models.analysis.__init__` carries a module docstring only. Callers
import the leaf modules.

## Interactions

`runs`, `pareto`, `plots`, and `report` operate on local run directories
written by `tools.ml_models.train.loop`. `cost` profiles torch modules
from `tools.ml_models.arch.registry`.

## Constraints

- `plots` and `report` consume the run artifacts written for a training
  run; they do not recompute metrics.
- Torch imports happen inside the modules that need them.

## Related documents

- [`tools.ml_models`](../ml_models.md)
- [`tools.ml_models.train.loop`](../train/loop.md)
- [`tools.ml_models.dataset`](../dataset.md)
- [`tools`](../tools.md)
