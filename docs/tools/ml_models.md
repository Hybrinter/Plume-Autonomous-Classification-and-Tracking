# tools.ml_models

**Source:** `packages/tools/src/tools/ml_models/`
**Kind:** package

## Purpose

The ml_models package holds finished-dataset builds and network builders
for model workflows.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`dataset`](ml_models/dataset.md) | package | Raw sources, finished-dataset build, manifest, and loader |
| [`arch`](ml_models/arch.md) | package | Segmentor and classifier network builders |
| [`train`](ml_models/train.md) | package | GSD-conditioned training loop, evaluation, and metrics |
| [`export`](ml_models/export.md) | package | Two-input ONNX export, manifests, acceptance, and pair gates |
| [`cli`](ml_models/cli.md) | module | `python -m tools.ml_models` dataset, train, export, accept, and pair commands |
| [`__main__`](ml_models/__main__.md) | module | `python -m tools.ml_models` entry shim |

## Package interface

`tools.ml_models.__init__` carries a module docstring only. Callers import
`tools.ml_models.dataset` and `tools.ml_models.arch`.

## Interactions

`tools.ml_models.dataset.preprocess` calls
`flight.payload.preprocess.normalize.normalize_dn` and
`flight.payload.gimbal.footprint` supplies the GSD encoding. The package
does not publish on the bus. `tools.ml_models.dataset` and
`tools.ml_models.arch` do not import `flight.payload.inference`,
`flight.core`, or `tools.analysis`. `tools.ml_models.arch` does not
import `flight`. `tools.ml_models.export` is the exception for
`flight.payload.inference`: it re-exports the conditioned-shape verifier
from `flight.payload.inference.contract` so the graph contract has one
implementation. `tools.ml_models.cli` calls
`tools.ml_models.dataset.build`, `tools.ml_models.train.loop`, and the
`tools.ml_models.export` modules. The
root tools CLI mounts `tools.ml_models.cli` as `ml-models`.

## Constraints

- Inside `tools.ml_models.dataset`, only `loader` imports torch.
- `tools.ml_models.arch` does not import `flight`.
- `tools.ml_models.train` imports torch and `flight.libs.types` for the
  `Result` boundary; the CLI loads it lazily.
- Dataset roots are local directories. This package does not fetch a
  corpus.
- The package `__init__` does not re-export names.

## Related documents

- [`tools`](../tools.md)
- [`tools.ml_models.dataset`](ml_models/dataset.md)
- [`tools.ml_models.arch`](ml_models/arch.md)
- [`tools.ml_models.cli`](ml_models/cli.md)
- [`flight.payload.preprocess.normalize`](../flight/payload/preprocess/normalize.md)
