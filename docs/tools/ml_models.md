# tools.ml_models

**Source:** `packages/tools/src/tools/ml_models/`
**Kind:** package

## Purpose

The ml_models package holds processed-pack data for model workflows.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`data`](ml_models/data.md) | package | Pack metadata, group splits, and normalization |

## Package interface

`tools.ml_models.__init__` carries a module docstring only. Callers import
`tools.ml_models.data`.

## Interactions

`tools.ml_models.data.norm` calls
`flight.payload.preprocess.normalize.normalize_dn`. The package does not publish
on the bus. It does not import `flight.payload.inference`, `flight.core`, or
`tools.analysis`.

## Constraints

- `tools.ml_models.data` does not import torch.
- Pack files are local directories. This package does not fetch a corpus.
- The package `__init__` does not re-export names.

## Related documents

- [`tools`](../tools.md)
- [`tools.ml_models.data`](ml_models/data.md)
- [`flight.payload.preprocess.normalize`](../flight/payload/preprocess/normalize.md)
