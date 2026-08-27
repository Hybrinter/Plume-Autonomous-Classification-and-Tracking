# tools.inference.split

**Source:** `packages/tools/src/tools/inference/split.py`
**Kind:** module

## Purpose

This module assigns frozen train, val, and test indices and hashes a processed
pack.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SplitRecipe` | class | Seed plus train/val/test fractions |
| `SplitIndex` | class | Integer indices for each split |
| `DatasetMeta` | class | Pack identity including `dataset_hash` |
| `load_split_recipe` | function | Parse the committed or overlay TOML |
| `assign_splits` | function | Seeded permutation into three splits |
| `write_splits` / `load_splits` | function | `splits.json` codec |
| `compute_dataset_hash` | function | SHA-256 over pack file digests |
| `write_dataset_meta` / `load_dataset_meta` | function | `dataset.json` codec |

## Inputs and outputs

`load_split_recipe(path=None) -> SplitRecipe`.

`assign_splits(n, recipe) -> SplitIndex`. `n` must be at least 3.

`compute_dataset_hash(pack_dir) -> str` lowercase hex.

`load_splits(path) -> SplitIndex`.

`load_dataset_meta(path) -> DatasetMeta`.

## Behavior

1. Load seed and fractions from TOML. Default fractions are 0.70 / 0.15 / 0.15.
2. Permute `range(n)` with the recipe seed.
3. Give at least one index to val and test. Send leftover indices to train.
4. Hash `images.npy`, `masks.npy`, `labels.npy`, and `splits.json` by file digest.

## Errors and faults

`ValueError` when fractions are not positive, do not sum to 1.0, `n` is below
3, or split indices overlap. `FileNotFoundError` when a pack file is missing.

## Messages

None.

## Configuration

Default recipe path is `data/manifests/zenodo_4250706_splits.toml`.

## Constraints

The recipe applies to sorted paired filenames after preprocess. The corpus
file list is not stored in git. This module does not import torch.

## Related documents

- [`tools.inference.data`](data.md)
- [`tools.inference.fetch`](fetch.md)
