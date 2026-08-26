# Industrial Smoke Plume Detection — Model Architecture

This document describes the **current 4-channel** model architecture. The
original 12-channel design from the 2020 NeurIPS paper is archived under
`docs/legacy/`.

## Inputs

- Sentinel-2 GeoTIFF chips (`.tif`), normalized to `120 × 120` spatial size.
- **Four channels**, in this order: `B2` (blue, 490 nm), `B3` (green, 560 nm),
  `B4` (red, 665 nm), `B8` (NIR, 842 nm).
- Per-channel normalization with dataset-level mean/std (see
  `smoke_detection.data.transforms.Normalize`).

## Classification

- Backbone: `torchvision.models.resnet50` with ImageNet weights.
- First conv replaced to accept 4 input channels.
- Final FC replaced with `Linear(2048, 1)` producing a single logit.
- Loss: `BCEWithLogitsLoss`. Metric: accuracy, AUC.
- Training augmentation: random flips, 90-degree rotations, random crop to
  `90 × 90`.

## Segmentation

- Backbone: a 4-channel U-Net (from milesial/Pytorch-UNet, GPL v3).
- Output: per-pixel logit map; binarized at threshold `0`.
- Loss: `BCEWithLogitsLoss`. Metrics: IoU, image-level accuracy, area ratio.
- Same augmentation as classification, applied consistently to image + mask.

## Training framework

Both tasks are trained with **PyTorch Lightning**. `LightningDataModule`s own
the data pipeline; `LightningModule`s own loss/optimizer/metrics; the CLI
(`src/smoke_detection/cli/train.py`) constructs a `Trainer` with
`TensorBoardLogger` + `ModelCheckpoint` and wires everything from a YAML
config.

See `configs/classification/default.yaml` and `configs/segmentation/default.yaml`
for the full set of hyperparameters.
