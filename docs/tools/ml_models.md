# tools.ml_models

**Source:** `packages/tools/src/tools/ml_models/`
**Kind:** package

## Purpose

The ml_models package holds processed-pack data and network builders for model
workflows.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`data`](ml_models/data.md) | package | Packs, Zenodo reads, prism proxy, and flight canvas |
| [`arch`](ml_models/arch.md) | package | Segmentor and classifier network builders |

## Package interface

`tools.ml_models.__init__` carries a module docstring only. Callers import
`tools.ml_models.data` and `tools.ml_models.arch`.

## Interactions

`tools.ml_models.data.norm` calls
`flight.payload.preprocess.normalize.normalize_dn`. The package does not publish
on the bus. `tools.ml_models.data` and `tools.ml_models.arch` do not import
`flight.payload.inference`, `flight.core`, or `tools.analysis`.
`tools.ml_models.arch` does not import `flight`.

## Constraints

- `tools.ml_models.data` does not import torch.
- `tools.ml_models.arch` does not import `flight`.
- Pack files are local directories. This package does not fetch a corpus.
- The package `__init__` does not re-export names.

## Related documents

- [`tools`](../tools.md)
- [`tools.ml_models.data`](ml_models/data.md)
- [`tools.ml_models.arch`](ml_models/arch.md)
- [`flight.payload.preprocess.normalize`](../flight/payload/preprocess/normalize.md)
