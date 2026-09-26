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
| [`loop`](train/loop.md) | module | Single-pack batches and mixed chip and tile steps |
| [`recipe`](train/recipe.md) | module | Index split recipe and the legacy pack hash |
| [`samples`](train/samples.md) | module | Synthetic scenes and torch pack loaders |
| [`tiles`](train/tiles.md) | module | Study tiles at one band subset and side |
| [`study`](train/study.md) | module | Shared AdamW loop for the band study |
| [`native_sweep`](train/native_sweep.md) | module | Native-resolution study sweep |
| [`losses`](train/losses.md) | module | BCE, Dice, and focal objectives |
| [`metrics`](train/metrics.md) | module | Classifier and segmentor scores |
| [`cost`](train/cost.md) | module | Parameter and FLOP counts |
| [`sweep`](train/sweep.md) | module | Cartesian search over train configs |

## Package interface

`tools.ml_models.train.__init__` carries a module docstring only. Callers import
each module by name.

## Interactions

`loop` calls `config`, `losses`, `metrics`, `cost`, and
`tools.ml_models.arch.registry.build`. A mixed-extent run calls
`tools.ml_models.data.prism.union_location_split` and
`tools.ml_models.data.pack.load_processed_pack`. A single-pack run calls
`tools.ml_models.train.samples`. `sweep` calls `loop.train` and
`tools.ml_models.analysis.eval`.
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
- [`tools.ml_models.train.recipe`](train/recipe.md)
- [`tools.ml_models.train.samples`](train/samples.md)
- [`tools.ml_models.train.tiles`](train/tiles.md)
- [`tools.ml_models.train.study`](train/study.md)
- [`tools.ml_models.train.native_sweep`](train/native_sweep.md)
- [`tools.ml_models.train.losses`](train/losses.md)
- [`tools.ml_models.train.metrics`](train/metrics.md)
- [`tools.ml_models.train.cost`](train/cost.md)
- [`tools.ml_models.train.sweep`](train/sweep.md)
- [`tools.ml_models.data.prism`](data/prism.md)
- [`tools.ml_models.data.canvas`](data/canvas.md)
