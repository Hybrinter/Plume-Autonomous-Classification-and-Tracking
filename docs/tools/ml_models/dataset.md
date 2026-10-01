# tools.ml_models.dataset

**Source:** `packages/tools/src/tools/ml_models/dataset/`
**Kind:** package

## Purpose

The dataset package builds finished datasets from raw tile sources and loads
them for model work. Each source writes its own dataset root: shards of
`.npy` arrays under `<task>/<split>/<H>x<W>` plus a `dataset.json` manifest.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`raw`](dataset/raw.md) | module | Raw tile contract: refs, tiles, and the source protocol |
| [`geometry`](dataset/geometry.md) | module | Flight frame, grid, and tile geometry |
| [`preprocess`](dataset/preprocess.md) | module | Unit-interval conversion, uint16 quantization, model GSD |
| [`augment`](dataset/augment.md) | module | Offline dihedral element names and application |
| [`split`](dataset/split.md) | module | Group-wise train, val, and test indices |
| [`store`](dataset/store.md) | module | Shard writer and per-file readers |
| [`manifest`](dataset/manifest.md) | module | `dataset.json` schema and content hash |
| [`spec`](dataset/spec.md) | module | `BuildSpec` TOML schema for a build |
| [`build`](dataset/build.md) | module | Build a finished dataset from any raw source |
| [`loader`](dataset/loader.md) | module | Torch datasets and seeded single-shard batches |
| [`sources`](dataset/sources.md) | package | Raw sources: flight tile directory, synthetic, and Zenodo 4250706 |

## Package interface

`tools.ml_models.dataset.__init__` carries a module docstring only. Callers
import each module by name.

## Interactions

`build` consumes a `RawSource` from `sources` (including the `zenodo`
archive adapter) and writes shards through
`store`, then writes `dataset.json` through `manifest`. `build` calls
`preprocess`, `augment`, `split`, and `geometry`. `preprocess` calls
`flight.payload.preprocess.normalize.normalize_dn`. `loader` reads
`manifest` and the shard files and returns torch tensors. No module
publishes on the bus.

## Constraints

- `loader` is the only module in this package that imports torch.
- Stored images are uint16 on the 65535 grid, shaped `(N, 3, H, W)`.
- Stored GSD is float32 metres, shaped `(N, 2)`. Labels are float32
  `(N, 1)`. Segmentor masks are uint8 `(N, 1, H, W)`.
- A build writes to a sibling `.<name>.partial-*` directory and renames it
  onto the destination only after `dataset.json` is in place.
- No module imports `flight.payload.inference`, `flight.core`, or
  `tools.analysis`.

## Related documents

- [`tools.ml_models`](../ml_models.md)
- [`tools.ml_models.dataset.build`](dataset/build.md)
- [`tools.ml_models.dataset.loader`](dataset/loader.md)
- [`tools.ml_models.dataset.manifest`](dataset/manifest.md)
- [`tools.ml_models.dataset.sources`](dataset/sources.md)
- [`flight.payload.preprocess.normalize`](../../flight/payload/preprocess/normalize.md)
