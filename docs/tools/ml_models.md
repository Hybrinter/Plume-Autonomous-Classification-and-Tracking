# tools.ml_models

**Source:** `packages/tools/src/tools/ml_models/`
**Kind:** package

## Purpose

The ml_models package holds processed-pack data, network builders, and the
plain-torch train loop.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`data`](ml_models/data.md) | package | Packs, Zenodo reads, prism proxy, and flight canvas |
| [`arch`](ml_models/arch.md) | package | Segmentor and classifier network builders |
| [`train`](ml_models/train.md) | package | Train loop, losses, metrics, cost, and sweeps |

## Package interface

`tools.ml_models.__init__` carries a module docstring only. Callers import
`tools.ml_models.data`, `tools.ml_models.arch`, and `tools.ml_models.train`.

## Interactions

`tools.ml_models.data.norm` calls
`flight.payload.preprocess.normalize.normalize_dn`. The package does not publish
on the bus. `tools.ml_models.data`, `tools.ml_models.arch`, and
`tools.ml_models.train` do not import `flight.payload.inference`,
`flight.core`, or `tools.analysis`. `tools.ml_models.arch` does not import
`flight`. `tools.ml_models.train.loop` calls `arch.registry.build`. A canvas
run calls `data.canvas.sample_view`.

## Constraints

- `tools.ml_models.data` does not import torch.
- `tools.ml_models.arch` does not import `flight`.
- `tools.ml_models.train` imports torch.
- Pack files are local directories. This package does not fetch a corpus.
- The package `__init__` does not re-export names.

## Related documents

- [`tools`](../tools.md)
- [`tools.ml_models.data`](ml_models/data.md)
- [`tools.ml_models.arch`](ml_models/arch.md)
- [`tools.ml_models.train`](ml_models/train.md)
- [`flight.payload.preprocess.normalize`](../flight/payload/preprocess/normalize.md)
