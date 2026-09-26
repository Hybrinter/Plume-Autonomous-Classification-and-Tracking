# tools.ml_models.train.loop

**Source:** `packages/tools/src/tools/ml_models/train/loop.py`
**Kind:** module

## Purpose

This module runs the plain-torch train loop and writes a run directory. With
`canvas` unset, batches come from a chip `DataLoader`. With `canvas` set, each
step builds a flight frame or a window.

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
`checkpoints/best.pt`, and `summary.json`. A canvas run also writes
`batch_shapes.json`.

## Behavior

1. Resolve the architecture name. Empty `arch` selects `pactnet` for the
   classifier and `dilatenet` for the segmentor. Empty `run_id` becomes
   `{kind}-{arch}-{seed}-{digest8}`.
2. Raise `FileExistsError` when the run directory already has `summary.json`
   and `overwrite` is false.
3. With `canvas` unset, load a processed pack, an unsplit disk adapter, or a
   synthetic pack. `in_channels` must equal the pack channel count.
4. Probe one chip step at `batch_size`. A CUDA out-of-memory error halves the
   size down to 1. The written `config.toml` stores the size that fitted.
5. Build the objective from `tools.ml_models.train.losses`. Run the optimizer
   for `epochs`. `max_steps` stops the optimizer early. `None` runs full epochs.
6. On a chip run, CUDA mixed precision runs when `amp` is true and the device
   is CUDA. cuDNN benchmark is enabled on CUDA.
7. Score unaugmented train and val chip splits every `eval_interval` epochs.
   The final epoch is always scored. The test split is not scored.
8. Write `last.pt` every scored epoch. Write `best.pt` when the val metric
   improves. Chip selection uses F1 for a classifier and mean IoU for a
   segmentor, unless `val_metric` names `f1`, `mean_iou`, or `bce`.
9. With `canvas` set, load the pack with `load_processed_pack`. Build chips
   from one split. A row is annotated when its mask has a positive pixel. A
   positive label with an all-zero mask stays unannotated.
10. Call `torch.manual_seed` with `TrainConfig.seed` and build a numpy
    `Generator` from `CanvasConfig.seed` before `registry.build`.
    `cudnn.benchmark` stays false.
11. Optimizer step `i` is a full frame when `i % full_frame_every == 0`. Other
    steps are windows from `sample_view`. The frame size is `canvas.frame_hw`.
    The window side is `canvas.window_px`.
12. The segmentor objective is `focal_dice`. The classifier objective is
    BCE-with-logits on the max logit. When the module has `spatial` and the
    sample mask has a polygon, `focal_dice` is added on that map. Each spatial
    cell is positive when any input pixel in the cell is positive. A module
    without `spatial` skips that term.
13. A canvas run on CUDA uses `torch.autocast` and `GradScaler`. A CPU canvas
    run does not enter autocast.
14. Validation uses full-frame samples from the val split. Checkpoint
    selection uses classifier F1 of the max logit, or segmentor Dice.
15. `best.pt` and `last.pt` store model state, epoch, `dataset_hash`, arch,
    `in_channels`, and `band_names`. A canvas checkpoint also stores `frame_hw`,
    `window_px`, `ingest_path`, and `radiometry`. `summary.json` stores the
    dataset hash, the val metric, `in_channels`, and `band_names`. A canvas
    summary also stores `frame_hw`, `window_px`, `ingest_path`, and
    `radiometry`. `history.csv` stores the scored rows. `train` does not take
    a test loader.

## Errors and faults

`ValueError` on an unknown kind, architecture, optimizer, scheduler, loss, or
empty train split. `ValueError` when `in_channels` disagrees with the pack, or
when a canvas pack's height or width disagrees with `chip_side`.
`FileExistsError` when the run directory exists and `overwrite` is false.
`RuntimeError` when a CUDA out-of-memory error persists at `batch_size` 1.

## Messages

None.

## Configuration

See [`tools.ml_models.train.config`](config.md). Canvas geometry comes from
`CanvasConfig`. The loop reads `frame_hw` from that object.

## Constraints

Torch is a required tools dependency. Device is CUDA when present, else CPU,
unless `device` is set. The test split is not scored. A canvas run does not
allocate a fixed 1544 by 2064 frame inside the loop.

## Related documents

- [`tools.ml_models.train`](../train.md)
- [`tools.ml_models.train.config`](config.md)
- [`tools.ml_models.train.losses`](losses.md)
- [`tools.ml_models.train.metrics`](metrics.md)
- [`tools.ml_models.train.cost`](cost.md)
- [`tools.ml_models.data.canvas`](../data/canvas.md)
- [`tools.ml_models.data.pack`](../data/pack.md)
- [`tools.ml_models.arch.registry`](../arch/registry.md)
