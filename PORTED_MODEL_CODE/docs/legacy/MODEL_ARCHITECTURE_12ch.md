> **LEGACY DOCUMENT — preserved for reference.** This describes the original
> 12-channel (Sentinel-2 bands 1–10, 12, 13) model from the NeurIPS 2020
> paper. The current repository ships a 4-channel (B2, B3, B4, B8) pipeline.
> See `docs/MODEL_ARCHITECTURE.md` for the current architecture.

# Industrial Smoke Plume Detection - Model Architecture

## Overview

This repository implements a two-stage computer vision system on Sentinel-2 multispectral satellite imagery:

1. **Classification stage** (`classification/`): predicts whether a smoke plume is present in an image patch.
2. **Segmentation stage** (`segmentation/`): predicts a per-pixel smoke mask for images, enabling area estimation.

Both stages operate on **12 Sentinel-2 channels** (bands 1-10, 12, 13; band 11 is intentionally omitted).

---

## End-to-End System Design

### Inputs
- GeoTIFF image chips (`.tif`) from Sentinel-2.
- Each sample is forced to a fixed spatial size of `120 x 120`.
- Channel order used in code is bands `[1,2,3,4,5,6,7,8,9,10,12,13]`.

### Shared Preprocessing Ideas
- **Band selection:** skip band 11 (cirrus) because it is considered not informative for Level-2A products.
- **Normalization:** each channel is normalized with fixed dataset-level means and standard deviations.
- **Augmentation (training):**
  - random orientation changes (horizontal/vertical flips + 90-degree rotations),
  - random crop from `120 x 120` to `90 x 90`.

### Outputs
- Classification output: one logit per image (`smoke` vs `no smoke`).
- Segmentation output: one logit map per image (`H x W`) converted to binary mask by thresholding at 0.

---

## Major Section 1: Classification Architecture

## 1.1 Model Backbone and Head

The classification model is defined in `classification/model.py` and is a modified **ResNet-50**:

- Base network: `torchvision.models.resnet50(pretrained=True)`.
- First convolution replaced to support multispectral input:
  - from 3 input channels to **12 channels**.
- Final fully connected layer replaced:
  - from 1000-class output to a single scalar logit (`Linear(2048, 1)`).

This makes the architecture suitable for binary smoke detection while keeping a strong pretrained backbone.

## 1.2 Classification Data Pipeline

Implemented in `classification/data.py`.

### Dataset construction
- Expects directory structure with `positive/` and `negative/` folders.
- Reads all `.tif` files and labels by folder name.
- Optional class balancing:
  - `upsample`: duplicate positive examples to match negatives.
  - `downsample`: subsample negatives to match positives.

### Transform stack (default training behavior)
1. `Normalize`
2. `RandomCrop` (`120 -> 90`)
3. `Randomize` (flip/mirror/rotate)
4. `ToTensor`

## 1.3 Training Setup

Implemented in `classification/train.py`.

- Loss: `BCEWithLogitsLoss`.
- Optimizer: SGD (`lr`, `momentum` configurable).
- Scheduler: `ReduceLROnPlateau`.
- Data loading:
  - random sampling with replacement (`num_samples = 2/3 of dataset length`).
  - separate train and validation dataloaders.
- Metrics:
  - image-level binary accuracy by thresholding logits at `0`.
- Logging:
  - TensorBoard scalars (loss, accuracy, learning rate).

## 1.4 Evaluation Behavior

Implemented in `classification/eval.py`.

- Loads trained weights into the modified ResNet-50.
- Computes test-set accuracy.
- Includes optional diagnostic visualization mode (`batch_size = 1`) showing:
  - RGB composite,
  - false-color spectral composite (aerosols/water vapor/SWIR),
  - intermediate feature activation maps via forward hooks.

## 1.5 Classification Strengths and Constraints

### Strengths
- Leverages pretrained deep backbone.
- Uses full multispectral context, not only RGB.
- Balancing logic addresses class imbalance.

### Constraints
- Paths in scripts are placeholders and must be edited manually.
- In `model.py`, replacing `conv1` discards pretrained RGB first-layer weights.
- Static normalization constants assume similar sensor/data distribution.

---

## Major Section 2: Segmentation Architecture

## 2.1 U-Net Architecture

Implemented in `segmentation/model_unet.py`.

The model is a standard U-Net variant:

- Input channels: **12**
- Output channels/classes: **1** (binary smoke mask logit)
- Encoder:
  - repeated `DoubleConv` blocks + max-pooling downsampling
  - channel progression: `64 -> 128 -> 256 -> 512 -> 1024/factor`
- Decoder:
  - bilinear upsampling (default) + skip connections + `DoubleConv`
- Final layer: `1x1 Conv` to produce per-pixel logits.

## 2.2 Segmentation Data Pipeline and Label Rasterization

Implemented in `segmentation/data.py`.

### Data sources
- Image chips from image directory.
- Polygon annotations from JSON files in segmentation label directory.

### Label processing
- Annotation polygons are read from JSON completions.
- Polygons are scaled by `1.2` (per code comment) before rasterization.
- `rasterio.features.rasterize` converts polygons into binary masks.

### Dataset balancing strategy
- Positive images are those with non-empty polygons.
- Negative images are added until count equals number of positives (1:1 class balance at image level).

### Transform stack
1. `Normalize`
2. `Randomize`
3. `RandomCrop` (`120 -> 90`, applied to image and mask consistently)
4. `ToTensor`

## 2.3 Training Setup

Implemented in `segmentation/train.py`.

- Loss: `BCEWithLogitsLoss` on pixel logits.
- Optimizer: SGD with momentum.
- Scheduler: `ReduceLROnPlateau` driven by validation loss.
- Sampling:
  - random sampling with replacement for train/val subsets.

### Metrics tracked
- **Pixel-mask IoU** (`jaccard_score`) for samples where both prediction and target contain smoke.
- **Image-level detection accuracy** by reducing masks to a smoke-present flag.
- **Area ratio** (`predicted smoke area / true smoke area`) with edge-case handling.
- Training and validation metrics logged to TensorBoard.

## 2.4 Evaluation Behavior

Implemented in `segmentation/eval.py`.

- Loads trained U-Net checkpoint.
- Computes:
  - IoU,
  - image-level accuracy,
  - mean predicted/true area ratio.
- Optional diagnostic output (`batch_size = 1`) overlays:
  - ground-truth mask (red),
  - predicted mask (green),
  - sample IoU annotation.

## 2.5 Segmentation Strengths and Constraints

### Strengths
- Pixel-level output supports plume extent estimation, not only detection.
- Skip connections preserve localization details.
- Unified multispectral processing pipeline with classification stage.

### Constraints
- Binary threshold at logit `>= 0` is fixed and not calibrated.
- Metrics exclude some empty-mask cases for IoU reporting by design.
- Data/label paths are script-local and require manual configuration.

---

## How the Two Stages Work Together

The architecture supports a practical workflow:

1. **Classify** image chips quickly to find likely smoke events.
2. **Segment** smoke pixels in relevant chips to estimate plume area and support downstream monitoring.

This decomposition reduces unnecessary dense prediction on obviously negative scenes while still enabling quantitative plume characterization when smoke is present.

---

## Key Files Reference

- `README.md`: project summary and reported benchmark values.
- `classification/model.py`: modified ResNet-50 definition.
- `classification/data.py`: classification dataset and transforms.
- `classification/train.py`: classification training loop.
- `classification/eval.py`: classification evaluation and diagnostics.
- `segmentation/model_unet.py`: U-Net definition.
- `segmentation/data.py`: segmentation dataset, polygon parsing, rasterization.
- `segmentation/train.py`: segmentation training loop and metrics.
- `segmentation/eval.py`: segmentation evaluation and visual diagnostics.
