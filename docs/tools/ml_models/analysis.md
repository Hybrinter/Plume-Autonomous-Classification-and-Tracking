# tools.ml_models.analysis

**Source:** `packages/tools/src/tools/ml_models/analysis/`
**Kind:** package

## Purpose

The analysis package writes figures, run catalogs, held-out scores, and
full-frame blob metrics.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`plots`](analysis/plots.md) | module | Headless PNG writers |
| [`report`](analysis/report.md) | module | Figures and `report.md` for one run |
| [`runs`](analysis/runs.md) | module | Run discovery and text tables |
| [`pareto`](analysis/pareto.md) | module | Parameter count against hit rate or val Dice |
| [`eval`](analysis/eval.md) | module | Checkpoint scoring on one split |
| [`results`](analysis/results.md) | module | Native and ground-sample markdown tables |
| [`native`](analysis/native.md) | module | Coarse logits on the 120 px mask |
| [`full_frame`](analysis/full_frame.md) | module | Gate, blob overlap, and canvas scenes |

## Package interface

`tools.ml_models.analysis.__init__` carries a module docstring only. Callers
import each module by name.

## Interactions

`plots` reads `tools.ml_models.train.metrics.sigmoid` and
`tools.ml_models.data.grid.gsd_m`. `eval` reads a processed pack through
`tools.inference.data` and rebuilds the network with
`tools.ml_models.arch.registry.build`. `full_frame` calls
`flight.payload.blobs.extract_blobs` and reads vision gates from
`flight.libs.config.PactConfig`. No module imports `flight.payload.inference`,
`flight.core`, or `tools.analysis`. No module publishes on the bus.

## Constraints

- The package imports torch in `eval` and `native`.
- `plots` selects the Agg backend on import.
- `full_frame` may import `flight.payload.blobs`.
- The package `__init__` does not re-export names.

## Related documents

- [`tools.ml_models`](../ml_models.md)
- [`tools.ml_models.analysis.plots`](analysis/plots.md)
- [`tools.ml_models.analysis.report`](analysis/report.md)
- [`tools.ml_models.analysis.runs`](analysis/runs.md)
- [`tools.ml_models.analysis.pareto`](analysis/pareto.md)
- [`tools.ml_models.analysis.eval`](analysis/eval.md)
- [`tools.ml_models.analysis.results`](analysis/results.md)
- [`tools.ml_models.analysis.native`](analysis/native.md)
- [`tools.ml_models.analysis.full_frame`](analysis/full_frame.md)
- [`flight.payload.blobs`](../../flight/payload/blobs.md)
