# tools.ml_models.export.calibration

**Source:** `packages/tools/src/tools/ml_models/export/calibration.py`
**Kind:** module

## Purpose

This module collects deterministic two-input calibration batches for
INT8 quantization from finished train shards. Each batch pairs one image
with its encoded GSD, matching the graph inputs exactly.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `calibration_batches` | function | Round-robin `{"image", "gsd"}` dicts across train shards |

## Inputs and outputs

`calibration_batches(dataset, model, samples=32)` takes one finished
dataset directory and a `ModelManifest`, and returns a list of float32
numpy dicts with `image` `(1, C, H, W)` and `gsd` `(1, 2)`.

## Behavior

1. Loads the `dataset.json` manifest.
2. Requires `band_names` and `gsd_reference_m` to match the model
   sidecar, and each train shard's spatial size to match the model
   `input_shape` when it declares fixed H/W.
3. Iterates `ShardDataset` rows for the model kind's train split in
   round-robin order across same-root size shards, using every row at
   most once and stopping at `samples` or when every shard is exhausted.

## Errors and faults

`ValueError` when `samples` is not positive, the dataset's preprocessing
differs from the model, a train shard size disagrees with a fixed model
input shape, or no train shard exists.
Manifest failures surface as `OSError` or `ValueError`.

## Messages

None.

## Configuration

`samples` bounds the batch count; the CLI exposes it as
`--calib-samples` (default 32).

## Constraints

- Calibration consumes train shards only, never held-out rows.
- Each batch carries both `image` and encoded `gsd` entries.

## Related documents

- [`tools.ml_models.export`](../export.md)
- [`tools.ml_models.export.precision`](precision.md)
- [`tools.ml_models.dataset.loader`](../dataset/loader.md)
