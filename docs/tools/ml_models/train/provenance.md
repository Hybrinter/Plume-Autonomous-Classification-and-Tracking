# tools.ml_models.train.provenance

**Source:** `packages/tools/src/tools/ml_models/train/provenance.py`
**Kind:** module

## Purpose

This module records the training-side dataset geometry and enforces that
each group stays inside one split across the dataset's task shards.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `training_provenance` | function | Provenance dict plus split-leakage validation |

## Inputs and outputs

`training_provenance(dataset, manifest, task) -> dict[str, object]`.
`dataset` is the one finished dataset root.

The returned dict carries `dataset` (path, source, source_ref,
dataset_hash, weight_table_id, bins), `train_samples`,
`spatial_shapes`, `gsd_bin_ids`, `gsd_min_m`, `gsd_max_m`,
`gsd_reference_m`, `band_names`, `in_channels`, and `norm` (`unit`).

## Behavior

1. Every manifest shard is read; the `rows.jsonl` count must equal the
   manifest record.
2. A `group_id` seen in two different splits — including across task
   shards — is rejected as leakage.
3. Train shards for the task must hold finite positive `(n, 2)` GSD
   arrays. Rows marked `gsd_nominal` count toward `train_samples` but are
   excluded from the recorded minima and maxima; at least one measured
   train row is required.
4. Spatial shapes and bin ids accumulate over the task's train shards
   only.

## Errors and faults

`ValueError` on an empty dataset path, a row count disagreement, a group
appearing in two splits, an invalid training GSD array, a task with no
training samples, or a train split with only nominal-GSD rows.

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
