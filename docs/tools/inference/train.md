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
| `train` | function | Run SGD + `BCEWithLogitsLoss` and write a run directory |

## Inputs and outputs

`load_train_config(path=None) -> TrainConfig`.

`overlay_train_config(cfg, ...) -> TrainConfig`.

`train(config=None) -> Path`. Returns the run directory.

The run directory holds `config.toml`, `history.csv`, `checkpoints/last.pt`,
`checkpoints/best.pt`, and `summary.json`.

## Behavior

1. Resolve architecture (`resnet50` or `unet`) and run id `{kind}-{arch}-{seed}`.
2. Load a processed pack, an unsplit disk adapter, or a synthetic pack.
3. Run SGD with `BCEWithLogitsLoss` for `epochs`. Shuffle is off.
4. After each epoch, score train and val splits and append `history.csv`.
5. Write `last.pt` every epoch. Write `best.pt` when the val metric improves.
6. Write `summary.json` with hashes, counts, and the best epoch.

## Errors and faults

`ValueError` on an unknown `kind`, architecture, or empty train split.

## Messages

None.

## Configuration

`TrainConfig` defaults: `kind=segmentor`, `input_height_px=256`,
`input_width_px=256`, `in_channels=4`, `epochs=1`, `batch_size=2`,
`learning_rate=0.01`, `momentum=0.9`, `weight_decay=0.0`,
`run_dir=artifacts/runs`. Classifier val metric defaults to F1. Segmentor val
metric defaults to mean IoU. A TOML file may overlay these fields.

## Constraints

Torch is a required tools dependency. Spatial size comes from config, not from
a hardcoded architecture constant. The train loader does not shuffle. Device is
CUDA when present, else CPU.

## Related documents

- [`tools.inference`](../inference.md)
- [`tools.inference.data`](data.md)
- [`tools.inference.metrics`](metrics.md)
- [`tools.inference.export`](export.md)
- [`tools.inference.arch.registry`](arch/registry.md)
