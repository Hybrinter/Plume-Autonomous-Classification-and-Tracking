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
| [`synthetic`](sources/synthetic.md) | module | In-memory planted-blob tiles at the flight size |

## Package interface

`tools.ml_models.dataset.sources.__init__` carries a module docstring
only. Callers import `flight` or `synthetic` by module name.

## Interactions

`build_dataset` accepts any `RawSource`. `build_flight` constructs a
`FlightTileDir` from a directory. `build_synthetic` constructs a
`SyntheticSource`. Both modules read geometry constants from
`tools.ml_models.dataset.geometry` and row types from
`tools.ml_models.dataset.raw`.

## Constraints

- No module in this package imports torch.
- A source yields each tile once, in `index` order.
- `flight` performs file I/O; `synthetic` does not touch disk.

## Related documents

- [`tools.ml_models.dataset`](../dataset.md)
- [`tools.ml_models.dataset.raw`](raw.md)
- [`tools.ml_models.dataset.build`](build.md)
- [`tools.ml_models.dataset.sources.flight`](sources/flight.md)
- [`tools.ml_models.dataset.sources.synthetic`](sources/synthetic.md)
