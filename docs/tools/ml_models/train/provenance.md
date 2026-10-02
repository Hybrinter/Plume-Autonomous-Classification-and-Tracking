# tools.ml_models.train.provenance

**Source:** `packages/tools/src/tools/ml_models/train/provenance.py`
**Kind:** module

## Purpose

This module records the training-side dataset geometry and enforces that
groups shared across dataset roots stay inside one split.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `training_provenance` | function | Provenance dict plus split-leakage validation |

## Inputs and outputs

`training_provenance(dests, manifests, task) -> dict[str, object]`.
`dests` and `manifests` align element-wise and must be nonempty.

The returned dict carries `datasets` (path, source, source_ref,
dataset_hash, weight_table_id, bins), `train_samples`,
`spatial_shapes`, `gsd_bin_ids`, `gsd_min_m`, `gsd_max_m`,
`gsd_reference_m`, `band_names`, `in_channels`, and `norm` (`unit`).

## Behavior

1. Every manifest shard for the task is read; the `rows.jsonl` count must
   equal the manifest record.
2. Rows sharing `(source, source_ref or resolved dest, group_id)` must
   land in one split across all datasets.
3. Train shards must hold finite positive `(n, 2)` GSD arrays; their
   minima and maxima bound the recorded coverage.
4. Spatial shapes and bin ids accumulate over train shards only.

## Errors and faults

`ValueError` on a misaligned or empty dataset list, a row count
disagreement, a group appearing in two splits, an invalid training GSD
array, or a task with no training samples.

## Messages

None.

## Configuration

None.

## Constraints

Reads `rows.jsonl` and `gsd.npy` per shard; does not decode images.

## Related documents

- [`tools.ml_models.train`](../train.md)
- [`tools.ml_models.dataset.manifest`](../dataset/manifest.md)
- [`tools.ml_models.dataset.store`](../dataset/store.md)
