# tools.ml_models.train

**Source:** `packages/tools/src/tools/ml_models/train/`
**Kind:** package

## Purpose

The train package holds training configuration, run provenance, losses,
and the imperative evidence-recording training loop for the
GSD-conditioned model families.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`config`](train/config.md) | module | Frozen `TrainConfig`, TOML load, overlay, digest, and metric aliases |
| [`loop`](train/loop.md) | module | Public train boundary; durable run evidence and checkpoints |
| [`provenance`](train/provenance.md) | module | Training geometry and split-leakage checks |
| [`losses`](train/losses.md) | module | BCE, Dice, and focal objectives |

## Package interface

`tools.ml_models.train.__init__` carries a module docstring only. Callers
import `tools.ml_models.train.loop` and `tools.ml_models.train.config`.

## Interactions

`loop.train` is the `Result[Path, str]` boundary the `ml-models train`
CLI command calls; `Ok` carries the run directory. `provenance` reads
finished-dataset manifests; `loop` evaluates through
`tools.ml_models.analysis.evaluate` and records through
`tools.ml_models.analysis.training`.

## Constraints

- The package returns `Result[Path, str]` at the public boundary.
- Torch, the architecture registry, the loader, losses, and the
  evaluator are imported lazily inside `loop.train`; importing the
  package never requires torch.

## Related documents

- [`tools.ml_models`](../ml_models.md)
- [`tools.ml_models.dataset`](../dataset.md)
- [`tools.ml_models.arch`](../arch.md)
- [`tools.ml_models.cli`](../cli.md)
