# tools.ml_models.dataset.manifest

**Source:** `packages/tools/src/tools/ml_models/dataset/manifest.py`
**Kind:** module

## Purpose

This module defines `dataset.json`, the identity file of a finished
dataset, and the content hash over its shard files.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SCHEMA_VERSION` | constant | Manifest schema version, 2 |
| `BinRecord` | class | One named GSD bin row |
| `ShardCount` | class | Row count for one task, split, and spatial size |
| `DatasetManifest` | class | Parsed `dataset.json` identity |
| `compute_dataset_hash` | function | SHA-256 over every file except `dataset.json` |
| `write_manifest` / `load_manifest` | function | `dataset.json` codec |
| `check_compatible` | function | Shared bands, norm, and GSD reference across manifests |
| `shard_dir` | function | `<dataset>/<task>/<split>/<H>x<W>` path |
| `parse_shard_size` | function | Parse an `<H>x<W>` directory name |

## Inputs and outputs

`write_manifest(path, manifest)` writes JSON with `indent=2`. The
`schema_version` field is stored under the JSON key `schema`.

`load_manifest(path, verify=True) -> DatasetManifest`. With `verify`, the
stored hash is compared against a fresh `compute_dataset_hash` of the
parent directory.

`compute_dataset_hash(dataset_dir) -> str` returns a lowercase hex digest.

`check_compatible(manifests)` returns None or raises.

`parse_shard_size(name) -> tuple[int, int] | None`.

## Behavior

1. The hash digests `relative_posix_path:file_sha256` lines in sorted path
   order. Each file is read in 8 MiB chunks. `dataset.json` is excluded.
2. `DatasetManifest` requires schema 2, `norm` `unit`, `image_dtype`
   `float32`, a non-empty band list, at least one shard count, finite
   ordered GSD min/max ranges, a finite positive `gsd_reference_m`, and a
   64-character `dataset_hash`. A schema other than 2 is rejected.
3. `ShardCount` requires `n` at least 1, a positive size, and
   `n_positive` in `0..n`.
4. `check_compatible` compares `band_names`, `norm`, and
   `gsd_reference_m` against the first manifest. Shard sizes and GSD
   values may differ.

## Errors and faults

`FileNotFoundError` when `compute_dataset_hash` finds no files.
`OSError` / `json.JSONDecodeError` on a missing or malformed file.
`ValueError` when the payload fails the schema, the `schema` key is
absent or a version other than 2, the recomputed hash differs, or
`check_compatible` finds a mismatch or an empty sequence.

## Messages

None.

## Configuration

`SCHEMA_VERSION` is 2. The schema uses pydantic dataclasses with extra
keys forbidden. `split` and `augment` reuse the `SplitRecipe` and
`AugmentRecipe` types.

## Constraints

This module does not import torch. The manifest records the split and
augment recipes that produced the shards.

## Related documents

- [`tools.ml_models.dataset`](../dataset.md)
- [`tools.ml_models.dataset.build`](build.md)
- [`tools.ml_models.dataset.loader`](loader.md)
- [`tools.ml_models.dataset.split`](split.md)
- [`tools.ml_models.dataset.spec`](spec.md)
