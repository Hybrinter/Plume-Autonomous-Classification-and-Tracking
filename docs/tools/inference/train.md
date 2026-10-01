# tools.inference.train

**Source:** `packages/tools/src/tools/inference/train.py`
**Kind:** module

## Purpose

This module re-exports the plain-torch train loop. The loop body lives in
[`tools.ml_models.train.loop`](../ml_models/train/loop.md). With `chip_dir` and
`tile_dir` empty, batches come from a `DataLoader` over `SplitDataset`. A
flight run sets both directories and trains on chip batches and tile batches.
Each job writes a run directory.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `TrainConfig` | class | Frozen train hyperparameters |
| `load_train_config` | function | Defaults plus optional TOML overlay |
| `overlay_train_config` | function | Apply CLI field overlays |
| `apply_train_mapping` | function | Overlay from a string-key mapping |
| `config_digest` | function | 8-hex identity of experiment fields |
| `resolve_train_channels` | function | Channel count shared by the config and the pack |
| `write_train_config_toml` | function | Write a TrainConfig table |
| `train` | function | Run the loop and write a run directory |
| `is_cuda_oom` | function | Detect a CUDA allocator failure |
| `next_batch_after_oom` | function | Halve a batch size, or raise at size 1 |
| `fit_batch_size` | function | Lower the batch until one training step fits |

## Inputs and outputs

`load_train_config(path=None) -> TrainConfig`.

`overlay_train_config(cfg, ...) -> TrainConfig`.

`apply_train_mapping(cfg, data) -> TrainConfig`.

`config_digest(cfg) -> str`.

`resolve_train_channels(cfg, pack) -> int`. Raises `ValueError` when pack metadata disagrees with the image tensor, or when `in_channels` disagrees with the pack.

`train(config=None) -> Path`. Returns the run directory.

`is_cuda_oom(exc) -> bool`.

`next_batch_after_oom(batch) -> int`. Raises `RuntimeError` at size 1.

`fit_batch_size(requested, attempt) -> int`.

The run directory holds `config.toml`, `history.csv`, `checkpoints/last.pt`,
`checkpoints/best.pt`, and `summary.json`.

## Behavior

1. Resolve architecture through the registry grammar. Empty `arch` selects
   `pactnet` for the classifier and `dilatenet` for the segmentor. Empty
   `run_id` becomes `{kind}-{arch}-{seed}-{digest8}`. A supplied `run_id` is
   used unchanged.
2. Raise `FileExistsError` when the run directory already has `summary.json`
   and `overwrite` is false.
3. Load a processed pack, an unsplit disk adapter, or a synthetic pack.
   `in_channels` must equal the pack channel count. A mismatch raises
   `ValueError`. A synthetic pack is built with `cfg.in_channels`.
4. Probe one training step at `batch_size`. A CUDA out-of-memory error halves
   the size and retries down to 1. The written `config.toml` stores the size
   that fitted.
5. Build the objective from `tools.inference.losses`. Run the selected
   optimizer for `epochs`. Train shuffle is off by default. Train-split flip and
   rotation run when `augment` is true.
6. On a single-pack run, CUDA mixed precision runs when `amp` is true and the
   device is CUDA. That loop uses `torch.amp.autocast` and `GradScaler`. cuDNN
   benchmark is enabled on CUDA for a single-pack run. A mixed run keeps cuDNN
   benchmark off. A mixed run on CUDA uses `torch.autocast` and `GradScaler`
   when `amp` is true. A CPU mixed run does not enter autocast.
7. Score unaugmented train and val splits every `eval_interval` epochs. The
   final epoch is always scored. A cosine schedule steps once per epoch.
8. Write `last.pt` every scored epoch. Write `best.pt` when the val metric
   improves. Stop early when `patience` scored epochs pass without improvement.
9. Write `summary.json` with hashes, counts, `n_params`, `flops`, `loss`, `amp`,
   `batch_size`, `stopped_early`, `train_seconds`, and the best epoch.
10. With `chip_dir` and `tile_dir` set, load both packs. The caller writes the
    tile pack with `write_tile_pack` before `train`. Rows come from
    `union_location_split`. Each step is a chip batch or a tile batch. BCE and
    focal are reduced per image, then multiplied by `chip_weight` or
    `tile_weight`. Validation writes chip scores and tile scores. The
    checkpoint metric is tile classifier F1 or tile segmentor Dice.
    `max_steps` caps optimizer steps. `None` runs full epochs. The test split
    is not scored. Checkpoints store model state, epoch, `dataset_hash`, arch,
    `in_channels`, `band_names`, `chip_hw`, `tile_hw`, `gsd_m`, `chip_weight`,
    and `tile_weight`. A mixed run also writes `batch_shapes.json`.

## Errors and faults

`ValueError` on an unknown `kind`, architecture, optimizer, scheduler, loss
name, or empty train split. Unknown `kind`, `optimizer`, and `scheduler` fail
at schema construction. `ValueError` when pack metadata disagrees with image
tensor channels, or when `in_channels` disagrees with the pack channel count.
A 4-band pack needs `in_channels = 4`.
`FileExistsError` when the run directory exists and `overwrite` is false.
`RuntimeError` when a CUDA out-of-memory error persists at `batch_size` 1.

## Messages

None.

## Configuration

`TrainConfig` defaults: `kind=segmentor`, `arch=""`, `input_height_px=256`,
`input_width_px=256`, `in_channels=3`, `epochs=1`, `batch_size=2`,
`learning_rate=0.01`, `momentum=0.9`, `weight_decay=0.0`, `optimizer=sgd`,
`scheduler=none`, `shuffle=false`, `pos_weight=0.0`, `augment=false`,
`loss=bce`, `focal_gamma=2.0`, `focal_alpha=0.25`, `amp=false`, `patience=0`,
`eval_interval=1`, `max_steps` unset, `chip_dir=""`, `tile_dir=""`,
`chip_weight=1.0`, `tile_weight=1.0`, `run_dir=artifacts/runs`,
`overwrite=false`. Classifier val metric defaults to F1. Segmentor val metric
defaults to mean IoU. A mixed run selects tile classifier F1 or tile segmentor
Dice. A TOML file may overlay these fields. `config_digest` omits `run_dir`,
`run_id`, `checkpoint_path`, and `overwrite`. `patience` at or below zero
disables early stop. `max_steps` at or above 1 caps optimizer steps. The 256 px
fields size a synthetic pack. A flight run sets `chip_dir` and `tile_dir`.

## Constraints

Torch is a required tools dependency. Spatial size comes from config, not from
a hardcoded architecture constant. Device is CUDA when present, else CPU.
Unknown `optimizer`, `scheduler`, or `loss` values raise `ValueError`.

## Related documents

- [`tools.inference`](../inference.md)
- [`tools.inference.losses`](losses.md)
- [`tools.inference.cost`](cost.md)
- [`tools.inference.data`](data.md)
- [`tools.inference.metrics`](metrics.md)
- [`tools.inference.export`](export.md)
- [`tools.ml_models.arch.registry`](../ml_models/arch/registry.md)
- [`tools.ml_models.train`](../ml_models/train.md)
- [`tools.ml_models.train.loop`](../ml_models/train/loop.md)
- [`tools.ml_models.data.prism`](../ml_models/data/prism.md)
- [`tools.inference.sweep`](sweep.md)
