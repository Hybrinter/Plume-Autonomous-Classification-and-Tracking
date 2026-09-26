# tools.ml_models.train

**Source:** `packages/tools/src/tools/ml_models/train/`
**Kind:** package

## Purpose

The train package runs a plain-torch loop for the classifier and the segmentor.
It holds the objective, the scores, the cost counters, and the sweep helper.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`config`](train/config.md) | module | Frozen hyperparameters and the channel-count rule |
| [`loop`](train/loop.md) | module | Chip batches and flight-frame canvas steps |
| [`losses`](train/losses.md) | module | BCE, Dice, and focal objectives |
| [`metrics`](train/metrics.md) | module | Classifier and segmentor scores |
| [`cost`](train/cost.md) | module | Parameter and FLOP counts |
| [`sweep`](train/sweep.md) | module | Cartesian search over train configs |

## Package interface

`tools.ml_models.train.__init__` carries a module docstring only. Callers import
each module by name.

## Interactions

`loop` calls `config`, `losses`, `metrics`, `cost`, and
`tools.ml_models.arch.registry.build`. A canvas run calls
`tools.ml_models.data.canvas.sample_view` and
`tools.ml_models.data.pack.load_processed_pack`. A chip run calls
`tools.inference.data`. `sweep` calls `loop.train` and `tools.inference.eval`.
No module publishes on the bus.

## Constraints

- The package imports torch.
- No module imports `flight.payload.inference`, `flight.core`, or
  `tools.analysis`.
- `train` returns the run directory path.
- The test split is not scored inside `train`.

## Related documents

- [`tools.ml_models`](../ml_models.md)
- [`tools.ml_models.train.config`](train/config.md)
- [`tools.ml_models.train.loop`](train/loop.md)
- [`tools.ml_models.train.losses`](train/losses.md)
- [`tools.ml_models.train.metrics`](train/metrics.md)
- [`tools.ml_models.train.cost`](train/cost.md)
- [`tools.ml_models.train.sweep`](train/sweep.md)
- [`tools.ml_models.data.canvas`](../ml_models/data/canvas.md)
- [`tools.inference.train`](../inference/train.md)
