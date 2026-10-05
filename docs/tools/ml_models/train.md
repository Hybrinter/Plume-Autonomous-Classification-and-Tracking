# tools.ml_models.train

**Source:** `packages/tools/src/tools/ml_models/train/`
**Kind:** package

## Purpose

The train package holds training configuration, run provenance, and
losses for the GSD-conditioned model families. The training boundary
itself is unavailable until the evidence training phase lands.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`config`](train/config.md) | module | Frozen `TrainConfig`, TOML load, overlay, and digest |
| [`loop`](train/loop.md) | module | Public train boundary (unavailable) |
| [`provenance`](train/provenance.md) | module | Training geometry and split-leakage checks |
| [`losses`](train/losses.md) | module | BCE, Dice, and focal objectives |

## Package interface

`tools.ml_models.train.__init__` carries a module docstring only. Callers
import `tools.ml_models.train.loop` and `tools.ml_models.train.config`.

## Interactions

`loop.train` is the `Result[Path, str]` boundary the `ml-models train`
CLI command calls; it currently returns an explicit unavailable error.
`provenance` reads finished-dataset manifests.

## Constraints

- The package returns `Result[Path, str]` at the public boundary.
- `train` creates no run directory, model, checkpoint, or summary while
  unavailable.

## Related documents

- [`tools.ml_models`](../ml_models.md)
- [`tools.ml_models.dataset`](../dataset.md)
- [`tools.ml_models.arch`](../arch.md)
- [`tools.ml_models.cli`](../cli.md)
