# tools.ml_models.analysis

**Source:** `packages/tools/src/tools/ml_models/analysis/`
**Kind:** package

## Purpose

The analysis package scores finished model artifacts and training runs:
parameter and FLOP accounting, run-directory summaries and ranking, Pareto
frontiers, report rendering, and the 64-tile full-frame flight evaluation.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`cost`](analysis/cost.md) | module | Parameter and FLOP counters |
| [`runs`](analysis/runs.md) | module | Run discovery, summaries, and rank tables |
| [`pareto`](analysis/pareto.md) | module | Cost-versus-quality frontier and knee selection |
| [`plots`](analysis/plots.md) | module | History, overlay, and failure figures |
| [`report`](analysis/report.md) | module | `report.md` and figure bundle writer |
| [`full_frame`](analysis/full_frame.md) | module | 64-tile conditioned frame evaluation |

## Package interface

`tools.ml_models.analysis.__init__` carries a module docstring only. Callers
import the leaf modules. The `frame-eval` command on
`tools.ml_models.cli` loads two conditioned checkpoints and writes the
report from `full_frame.evaluate_flight_frames`.

## Interactions

`full_frame` reads finished flight datasets through
`tools.ml_models.dataset` helpers, uses flight's shared GSD, tiling and tile
inference functions, and scores results with `train.metrics`.
`runs`, `pareto`, `plots`, and `report` operate on local run directories
written by `tools.ml_models.train.loop`. `cost` profiles torch modules
from `tools.ml_models.arch.registry`.

## Constraints

- `full_frame` scores only complete 64-tile test frames; unannotated masks
  stay unknown.
- `plots` and `report` consume the run artifacts written for a training
  run; they do not recompute metrics.
- Torch imports happen inside the modules that need them.

## Related documents

- [`tools.ml_models`](../ml_models.md)
- [`tools.ml_models.train.loop`](../train/loop.md)
- [`tools.ml_models.dataset`](../dataset.md)
- [`tools`](../tools.md)
