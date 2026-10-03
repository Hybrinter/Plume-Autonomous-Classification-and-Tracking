# tools.ml_models.dataset.sources

**Source:** `packages/tools/src/tools/ml_models/dataset/sources/`
**Kind:** package

## Purpose

The sources package holds the raw tile producers a dataset build consumes.
Each module implements the `RawSource` protocol for one origin of tiles.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`flight`](sources/flight.md) | module | Labeled flight tile directory, read and write |
| [`zenodo`](sources/zenodo.md) | package | Zenodo 4250706 archive source at multiple GSD bins |

## Package interface

`tools.ml_models.dataset.sources.__init__` carries a module docstring
only. Callers import `flight` or `zenodo` by module name.

## Interactions

`build_dataset` accepts any `RawSource`. `build_flight` constructs a
`FlightTileDir` from a directory. `build_zenodo` constructs a
`ZenodoSource` from two archives and a weight table. These modules read
row types from `tools.ml_models.dataset.raw`; `zenodo` reads band
constants from `tools.ml_models.dataset.geometry`, and `flight` records
header defaults from flight `InferenceConfig`.

## Constraints

- No module in this package imports torch.
- A source yields each tile once, in `index` order.
- `flight` performs file I/O; `zenodo` streams archives without
  extraction.

## Related documents

- [`tools.ml_models.dataset`](../dataset.md)
- [`tools.ml_models.dataset.raw`](raw.md)
- [`tools.ml_models.dataset.build`](build.md)
- [`tools.ml_models.dataset.sources.flight`](sources/flight.md)
- [`tools.ml_models.dataset.sources.zenodo`](sources/zenodo.md)
