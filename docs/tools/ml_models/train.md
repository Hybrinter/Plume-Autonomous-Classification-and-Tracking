# tools.ml_models.train

**Source:** `packages/tools/src/tools/ml_models/train/`
**Kind:** package

## Purpose

The train package runs a plain-torch loop over finished datasets for the
GSD-conditioned model families. It covers configuration, run provenance,
evaluation, objectives, and metrics.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`config`](train/config.md) | module | Frozen `TrainConfig`, TOML load, overlay, and digest |
| [`loop`](train/loop.md) | module | Run directory, epoch loop, checkpoints, and artifacts |
| [`provenance`](train/provenance.md) | module | Training geometry and split-leakage checks |
| [`evaluate`](train/evaluate.md) | module | Exhaustive split scoring for one dataset |
| [`losses`](train/losses.md) | module | BCE, Dice, and focal objectives |
| [`metrics`](train/metrics.md) | module | Classifier and segmentor scores |

## Package interface

`tools.ml_models.train.__init__` carries a module docstring only. Callers
import `tools.ml_models.train.loop` and `tools.ml_models.train.config`.

## Interactions

`loop.train` reads finished datasets through
`tools.ml_models.dataset.loader` and `tools.ml_models.dataset.store`,
builds models through `tools.ml_models.arch.registry`, and scores
validation splits with `tools.ml_models.train.evaluate`. The
`ml-models train` CLI command in `tools.ml_models.cli` calls
`loop.train`.

## Constraints

- The package imports torch and returns `Result[Path, str]` at the public
  boundary.
- Each run trains on exactly one finished dataset.
- Checkpoints record `film-log-gsd-v1` or `ignored` conditioning.

## Related documents

- [`tools.ml_models`](../ml_models.md)
- [`tools.ml_models.dataset`](../dataset.md)
- [`tools.ml_models.arch`](../arch.md)
- [`tools.ml_models.cli`](../cli.md)
