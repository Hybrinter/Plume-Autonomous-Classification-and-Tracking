# tools.model.train

**Source:** `packages/tools/src/tools/model/train.py`
**Kind:** module

## Purpose

This module runs a plain-torch SGD loop for the classifier or the segmentor.
Importing the module does not import torch.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `TrainConfig` | class | Frozen train hyperparameters |
| `load_train_config` | function | Defaults plus optional TOML overlay |
| `overlay_train_config` | function | Apply CLI field overlays |
| `train` | function | Run SGD + `BCEWithLogitsLoss` and write a checkpoint |

## Inputs and outputs

`load_train_config(path=None) -> TrainConfig`.

`overlay_train_config(cfg, ...) -> TrainConfig`.

`train(config=None) -> Path`. Returns the checkpoint path. Raises `ImportError`
when torch is missing.

## Behavior

1. Load samples from `data_dir` or from the synthetic planted-blob generator.
2. Build a ResNet-50 classifier or a U-Net segmentor.
3. Run SGD with `BCEWithLogitsLoss` for `epochs` batches.
4. Write a checkpoint dict with `kind`, `state_dict`, and input geometry.

## Errors and faults

`ImportError` when torch is not installed. `ValueError` on an unknown `kind`.

## Messages

None.

## Configuration

`TrainConfig` defaults: `kind=segmentor`, `input_height_px=256`,
`input_width_px=256`, `in_channels=4`, `epochs=1`, `batch_size=2`,
`learning_rate=0.01`, `momentum=0.9`. A TOML file may overlay these fields.

## Constraints

Torch imports are lazy inside `train`. Spatial size comes from config, not from
a hardcoded architecture constant. The default `pact-tools` extra set does not
install torch.

## Related documents

- [`tools.model`](../model.md)
- [`tools.model.data`](data.md)
- [`tools.model.export`](export.md)
- [`tools.model.arch`](arch.md)
