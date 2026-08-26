# Augmentation Improvements

## Overview

Training-side augmentation improvements applied to both the classification and segmentation pipelines. No changes to model architecture or inference — these only affect what the model sees during training.

---

## Changes

### 1. Fixed augmentation order (classification only)

**File:** `src/smoke_detection/classification/data.py`

`RandomCrop` was previously applied before `Randomize`, meaning the model always saw the same cropped region in different orientations. Swapping the order means flips and rotations are applied to the full 120×120 image first, then a random 90×90 region is cropped — giving more spatial diversity per epoch.

| | Pipeline |
|---|---|
| Before | `Normalize → RandomCrop → Randomize → ToTensor` |
| After | `Normalize → SpectralJitter → GaussianNoise → Randomize → RandomCrop → ToTensor` |

The segmentation pipeline order was already correct and unchanged.

---

### 2. Added `SpectralJitter` transform

**Files:** `src/smoke_detection/classification/data.py`, `src/smoke_detection/segmentation/data.py`

Multiplies each of the 4 spectral channels independently by a random scalar drawn from `U(0.9, 1.1)`. Simulates real-world variation in atmospheric haze, illumination, and seasonal differences across acquisitions.

```python
class SpectralJitter(object):
    def __init__(self, low=0.9, high=1.1):
        self.low = low
        self.high = high

    def __call__(self, sample):
        factors = np.random.uniform(self.low, self.high, size=(sample['img'].shape[0], 1, 1))
        sample['img'] = sample['img'] * factors
        return sample
```

Applied after `Normalize`, before `GaussianNoise`. Only modifies `img`; labels and masks are unaffected.

---

### 3. Added `GaussianNoise` transform

**Files:** `src/smoke_detection/classification/data.py`, `src/smoke_detection/segmentation/data.py`

Adds zero-mean Gaussian noise (`std=0.05`) to each pixel. Simulates sensor readout noise and quantization variation. With Z-score normalized data sitting roughly in `[-3, 3]`, `std=0.05` is small enough not to corrupt the signal.

```python
class GaussianNoise(object):
    def __init__(self, std=0.05):
        self.std = std

    def __call__(self, sample):
        noise = np.random.normal(0, self.std, sample['img'].shape)
        sample['img'] = sample['img'] + noise
        return sample
```

Applied after `SpectralJitter`, before `Randomize`. Only modifies `img`; labels and masks are unaffected.

---

## Final pipeline (both tasks)

```
Normalize → SpectralJitter → GaussianNoise → Randomize → RandomCrop → ToTensor
```

---

## Inference impact

None. These transforms are only applied when `apply_transforms=True` in `create_dataset()`, which is used during training. Evaluation and inference call `create_dataset(..., apply_transforms=False)`.
