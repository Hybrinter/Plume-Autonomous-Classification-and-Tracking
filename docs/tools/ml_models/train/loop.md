# tools.ml_models.train.loop

**Source:** `packages/tools/src/tools/ml_models/train/loop.py`
**Kind:** module

## Purpose

This module runs the plain-torch train loop and writes a run directory. With
`chip_dir` and `tile_dir` empty, batches come from one pack `DataLoader`. With
both directories set, each step is a chip batch or a tile batch.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `train` | function | Run the loop and write a run directory |
| `is_cuda_oom` | function | Detect a CUDA allocator failure |
| `next_batch_after_oom` | function | Halve a batch size, or raise at size 1 |
| `fit_batch_size` | function | Lower the batch until one training step fits |

## Inputs and outputs

`train(config=None) -> Path`. The return value is the run directory.
`config` defaults to `TrainConfig()`.

`is_cuda_oom(exc) -> bool`.

`next_batch_after_oom(batch) -> int`. Raises `RuntimeError` at size 1.

`fit_batch_size(requested, attempt) -> int`.

The run directory holds `config.toml`, `history.csv`, `checkpoints/last.pt`,
`checkpoints/best.pt`, and `summary.json`. A mixed run also writes
`batch_shapes.json`.

## Behavior

1. Resolve the architecture name. Empty `arch` selects `pactnet` for the
   classifier and `dilatenet` for the segmentor. Empty `run_id` becomes
   `{kind}-{arch}-{seed}-{digest8}`.
2. Raise `FileExistsError` when the run directory already has `summary.json`
   and `overwrite` is false.
3. With `chip_dir` and `tile_dir` empty, load a processed pack, an unsplit
   disk adapter, or a synthetic pack. `in_channels` must equal the pack
   channel count. The 256 px fields size that synthetic pack.
4. Probe one step at `batch_size`. A CUDA out-of-memory error halves the size
   down to 1. The written `config.toml` stores the size that fitted.
5. On a single-pack run, build the objective from
   `tools.ml_models.train.losses`. Run the optimizer for `epochs`.
   `max_steps` stops the optimizer early. `None` runs full epochs.
6. On a single-pack run, CUDA mixed precision runs when `amp` is true and the
   device is CUDA. cuDNN benchmark is enabled on CUDA.
7. Score unaugmented train and val splits every `eval_interval` epochs. The
   final epoch is always scored. The test split is not scored.
8. Write `last.pt` every scored epoch. Write `best.pt` when the val metric
   improves. Single-pack selection uses F1 for a classifier and mean IoU for
   a segmentor, unless `val_metric` names `f1`, `mean_iou`, or `bce`.
9. With `chip_dir` and `tile_dir` set, load both processed packs. The caller
   writes the tile pack with `write_tile_pack` before `train`. The loop loads
   the two directories.
10. Index rows with `union_location_split` and `SplitRecipe(seed=seed)`. One
    location has one split on both packs. Pack `splits.json` indices are not
    the training index. The two packs stay separate.
11. Each optimizer step is one source. Chip batches and tile batches alternate.
    A chip batch is first when both sources have rows. A batch holds one
    spatial size.
12. The segmentor objective is per-image `focal_dice`, then the source weight.
    The classifier objective is per-image BCE-with-logits on the max logit.
    When the module has `spatial` and the sample mask has a polygon,
    per-image `focal_dice` is added on that map. Each spatial cell is positive
    when any input pixel in the cell is positive. A module without `spatial`
    skips that term.
13. BCE and focal use the mean inside each image before the source weight.
    Dice is already one value per image. `chip_weight` scales a chip batch.
    `tile_weight` scales a tile batch.
14. A mixed run on CUDA uses `torch.autocast` and `GradScaler` when `amp` is
    true. cuDNN benchmark stays false. A CPU mixed run does not enter autocast.
15. Validation writes chip scores and tile scores in `history.csv`. The
    `source` column is `chip` or `tile`. Checkpoint selection uses the tile
    metric: classifier F1 of the max logit, or segmentor Dice. An empty tile
    val split selects on the tile train score.
16. `best.pt` and `last.pt` store model state, epoch, the tile `dataset_hash`,
    arch, `in_channels`, and `band_names`. A mixed checkpoint also stores
    `chip_hw`, `tile_hw`, `gsd_m`, `chip_weight`, `tile_weight`, and
    `chip_dataset_hash`. `input_height_px` and `input_width_px` equal
    `tile_hw`. `summary.json` stores the same sizes, the ground sample
    distance, and the source weights.

## Errors and faults

`ValueError` on an unknown kind, architecture, optimizer, scheduler, loss, or
empty train split. `ValueError` when `in_channels` disagrees with a pack, when
one of `chip_dir` and `tile_dir` is empty, when `data_dir` is set on a mixed
run, or when a source weight is not finite and above zero.
`FileExistsError` when the run directory exists and `overwrite` is false.
`RuntimeError` when a CUDA out-of-memory error persists at `batch_size` 1.

## Messages

None.

## Configuration

See [`tools.ml_models.train.config`](config.md). A flight run sets `chip_dir`
and `tile_dir`. `chip_weight` and `tile_weight` scale the per-image loss.

## Constraints

Torch is a required tools dependency. Device is CUDA when present, else CPU,
unless `device` is set. The test split is not scored. A mixed run keeps chip
rows and tile rows in separate tensors.

## Related documents

- [`tools.ml_models.train`](../train.md)
- [`tools.ml_models.train.config`](config.md)
- [`tools.ml_models.train.losses`](losses.md)
- [`tools.ml_models.train.metrics`](metrics.md)
- [`tools.ml_models.train.cost`](cost.md)
- [`tools.ml_models.data.prism`](../data/prism.md)
- [`tools.ml_models.data.pack`](../data/pack.md)
- [`tools.ml_models.arch.registry`](../arch/registry.md)
