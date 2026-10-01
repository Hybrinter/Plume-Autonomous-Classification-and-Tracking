# tools.ml_models.data

**Source:** `packages/tools/src/tools/ml_models/data/`
**Kind:** package

## Purpose

The data package reads and writes processed packs: image tensors, masks,
labels, splits, and provenance.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`meta`](data/meta.md) | module | Dataset identity, provenance, and the pack hash |
| [`pack`](data/pack.md) | module | On-disk pack and in-memory concatenation |
| [`split`](data/split.md) | module | Group-wise train, val, and test indices |
| [`norm`](data/norm.md) | module | DN, unit-interval, and per-band z-score recipes |

## Package interface

`tools.ml_models.data.__init__` carries a module docstring only. Callers import
each module by name.

## Interactions

`norm` calls `flight.payload.preprocess.normalize.normalize_dn`. `pack` calls
`meta` and `split` to write sidecars and to assign groups. No module publishes
on the bus.

## Constraints

- No module in this package imports torch.
- No module imports `flight.payload.inference`, `flight.core`, or
  `tools.analysis`.
- Arrays on disk are float32. Images are `(N, C, H, W)`. Masks are
  `(N, 1, H, W)`. Labels are `(N, 1)`.

## Related documents

- [`tools.ml_models`](../ml_models.md)
- [`tools.ml_models.data.meta`](data/meta.md)
- [`tools.ml_models.data.pack`](data/pack.md)
- [`tools.ml_models.data.split`](data/split.md)
- [`tools.ml_models.data.norm`](data/norm.md)
