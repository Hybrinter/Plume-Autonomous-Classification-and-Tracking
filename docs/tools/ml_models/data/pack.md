# tools.ml_models.data.pack

**Source:** `packages/tools/src/tools/ml_models/data/pack.py`
**Kind:** module

## Purpose

This module writes and loads a processed pack, and stacks packs in memory.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `ProcessedPack` | class | Arrays, `SplitIndex`, and `DatasetMeta` |
| `write_processed_pack` | function | Write the six pack files and return `DatasetMeta` |
| `load_processed_pack` | function | Load a pack and check the hash and shapes |
| `assert_same_ingest` | function | Raise when `ingest_path` differs |
| `concat_packs` | function | Stack packs on N and remap split indices |

## Inputs and outputs

`write_processed_pack(...) -> DatasetMeta`.

Keyword arguments are `group_ids`, `splits`, and `recipe`. Positional arguments
are the destination, images, masks, labels, provenance, and source DOI.

Images are float32 `(N, C, H, W)`. Masks are float32 `(N, 1, H, W)`. Labels are
float32 `(N, 1)`. `C` equals `len(provenance.band_names)`.

Pass `group_ids` or `splits`, and not both. `len(group_ids)` equals N.

`load_processed_pack(dest) -> ProcessedPack`. Image, mask, and label arrays are
read-only memmaps.

`concat_packs(packs) -> ProcessedPack`. The result `n` is the sum of the input
counts. `dataset_hash` is an empty string.

## Behavior

1. Check dtypes and shapes. `C` equals the provenance band count.
2. With `group_ids`, call `assign_group_splits`. With `splits`, store that
   index. The index covers `0..N-1` with no duplicate.
3. Write `images.npy`, `masks.npy`, `labels.npy`, and `splits.json`.
4. Write `provenance.json`, compute the pack hash, then write `dataset.json`.
   Provenance fields are copied onto the meta.
5. `load_processed_pack` recomputes the hash. It checks
   `images.shape[1] == meta.in_channels == len(band_names)`, spatial size, mask
   shape `(N, 1, H, W)`, and label shape `(N, 1)`.
6. `concat_packs` calls `assert_same_ingest`. It then requires equal band
   names, spatial size, norm, and the rest of the provenance. Split indices
   shift by each preceding pack's N.
7. `concat_packs` does not write a directory.

## Errors and faults

`ValueError` when a shape, dtype, or split source is wrong, when the hash does
not match, when `ingest_path` differs, or when band names, spatial size, or
norm differ. `FileNotFoundError` when a pack file is missing.

## Messages

None.

## Configuration

`recipe` defaults to fractions 0.70 / 0.15 / 0.15 and seed 0 when `group_ids`
is passed and `recipe` is omitted. There is no TOML file.

## Constraints

Arrays are float32. The concatenated pack keeps the shared provenance. Its
`dataset_hash` stays empty until `write_processed_pack` writes the files.

## Related documents

- [`tools.ml_models.data`](../data.md)
- [`tools.ml_models.data.meta`](meta.md)
- [`tools.ml_models.data.split`](split.md)
