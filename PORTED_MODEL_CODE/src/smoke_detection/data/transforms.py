"""Shared transforms for classification and segmentation samples.

Samples are dicts: classification produces ``{'idx', 'img', 'lbl', 'imgfile'}``
and segmentation produces ``{'idx', 'img', 'fpt', 'imgfile'}``.
The transforms below inspect which keys are present and act accordingly.
"""

from __future__ import annotations

import numpy as np
import torch

# Dataset-level statistics for Sentinel-2 bands B2, B3, B4, B8.
CHANNEL_MEANS = np.array([900.5, 1061.4, 1091.7, 2186.3], dtype=np.float32)
CHANNEL_STDS = np.array([624.7, 640.8, 718.1, 947.9], dtype=np.float32)


class Normalize:
    """Per-channel normalization with fixed dataset statistics."""

    def __init__(self, means: np.ndarray = CHANNEL_MEANS, stds: np.ndarray = CHANNEL_STDS):
        self.means = means.reshape(-1, 1, 1)
        self.stds = stds.reshape(-1, 1, 1)

    def __call__(self, sample: dict) -> dict:
        sample["img"] = (sample["img"] - self.means) / self.stds
        return sample


class Randomize:
    """Random horizontal/vertical flips and 90-degree rotations.

    Applied consistently to the image and to the segmentation mask, if one is
    present in the sample under the ``fpt`` key.
    """

    def __call__(self, sample: dict) -> dict:
        img = sample["img"]
        fpt = sample.get("fpt")

        if np.random.randint(0, 2):
            img = np.flip(img, 2)
            if fpt is not None:
                fpt = np.flip(fpt, 1)
        if np.random.randint(0, 2):
            img = np.flip(img, 1)
            if fpt is not None:
                fpt = np.flip(fpt, 0)
        rot = np.random.randint(0, 4)
        img = np.rot90(img, rot, axes=(1, 2))
        if fpt is not None:
            fpt = np.rot90(fpt, rot, axes=(0, 1))

        sample["img"] = img.copy()
        if fpt is not None:
            sample["fpt"] = fpt.copy()
        return sample


class RandomCrop:
    """Random crop from 120x120 to a square of size ``crop``."""

    def __init__(self, crop: int = 90):
        self.crop = crop

    def __call__(self, sample: dict) -> dict:
        max_offset = 120 - self.crop
        x, y = np.random.randint(0, max_offset + 1, 2)
        sample["img"] = sample["img"][:, y : y + self.crop, x : x + self.crop].copy()
        if "fpt" in sample:
            sample["fpt"] = sample["fpt"][y : y + self.crop, x : x + self.crop].copy()
        return sample


class ToTensor:
    """Convert the ``img`` (and ``fpt`` if present) ndarray to ``torch.Tensor``."""

    def __call__(self, sample: dict) -> dict:
        sample["img"] = torch.from_numpy(np.ascontiguousarray(sample["img"])).float()
        if "fpt" in sample:
            sample["fpt"] = torch.from_numpy(np.ascontiguousarray(sample["fpt"])).float()
        return sample
