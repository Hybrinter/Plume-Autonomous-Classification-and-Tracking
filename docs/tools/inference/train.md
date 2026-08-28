# tools.inference.train

**Source:** `packages/tools/src/tools/inference/train.py`
**Kind:** module

## Purpose

This module runs a plain-torch SGD loop for the classifier or the segmentor.
Batches come from a `DataLoader` over `SplitDataset`. Each job writes a run
directory.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `TrainConfig` | class | Frozen train hyperparameters |
| `load_train_config` | function | Defaults plus optional TOML overlay |
| `overlay_train_config` | function | Apply CLI field overlays |
| `config_digest` | function | 8-hex identity of experiment fields |
| `train` | function | Run SGD + `BCEWithLogitsLoss` and write a run directory |

## Inputs and outputs

`load_train_config(path=None) -> TrainConfig`.

`overlay_train_config(cfg, ...) -> TrainConfig`.

`config_digest(cfg) -> str`.

`train(config=None) -> Path`. Returns the run directory.

The run directory holds `config.toml`, `history.csv`, `checkpoints/last.pt`,
`checkpoints/best.pt`, and `summary.json`.

## Behavior

1. Resolve architecture (`resnet50` or `unet`). Empty `run_id` becomes
   `{kind}-{arch}-{seed}-{digest8}`. A supplied `run_id` is used unchanged.
2. Raise `FileExistsError` when the run directory already has `summary.json`
   and `overwrite` is false.
3. Load a processed pack, an unsplit disk adapter, or a synthetic pack.
4. Run the selected optimizer with `BCEWithLogitsLoss` for `epochs`. Train
   shuffle is off by default. Train-split flip and rotation run when `augment`
   is true.
5. After each epoch, score unaugmented train and val splits and append
   `history.csv`. A cosine schedule steps once per epoch.
6. Write `last.pt` every epoch. Write `best.pt` when the val metric improves.
7. Write `summary.json` with hashes, counts, `n_params`, `flops`, and the best
   epoch.

## Errors and faults

`ValueError` on an unknown `kind`, architecture, optimizer, scheduler, or empty
train split. `FileExistsError` when the run directory exists and `overwrite` is
false.

## Messages

None.

## Configuration

`TrainConfig` defaults: `kind=segmentor`, `input_height_px=256`,
`input_width_px=256`, `in_channels=4`, `epochs=1`, `batch_size=2`,
`learning_rate=0.01`, `momentum=0.9`, `weight_decay=0.0`, `optimizer=sgd`,
`scheduler=none`, `shuffle=false`, `pos_weight=0.0`, `augment=false`,
`run_dir=artifacts/runs`, `overwrite=false`. Classifier val metric defaults to
F1. Segmentor val metric defaults to mean IoU. A TOML file may overlay these
fields. `config_digest` omits `run_dir`, `run_id`, `checkpoint_path`, and
`overwrite`.

## Constraints

Torch is a required tools dependency. Spatial size comes from config, not from
a hardcoded architecture constant. Device is CUDA when present, else CPU.
Unknown `optimizer` or `scheduler` values raise `ValueError`.

## Related documents

- [`tools.inference`](../inference.md)
- [`tools.inference.cost`](cost.md)
- [`tools.inference.data`](data.md)
- [`tools.inference.metrics`](metrics.md)
- [`tools.inference.export`](export.md)
- [`tools.inference.arch.registry`](arch/registry.md)
