"""Training tensors: synthetic scenes and an on-disk numpy adapter.

This module stays torch-free. Images are float32 in the flight `normalize_dn`
domain unless the disk loader sees DN-valued arrays and rescales them.

Contains:
  - SampleBatch: one image tensor plus classifier or segmentor targets.
  - make_synthetic_batch: planted-blob scenes for a 1-step train smoke test.
  - load_disk_batch: read `images.npy` plus `labels.npy` or `masks.npy`.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from flight.payload.preprocess import normalize_dn


@dataclass(frozen=True, slots=True)
class SampleBatch:
    """One training batch in numpy.

    Attributes:
        images: np.ndarray[float32, (N, C, H, W)] in [0, 1].
        targets: Classifier (N, 1) float32 labels, or segmentor (N, 1, H, W)
            float32 masks in {0, 1}.
    """

    images: np.ndarray
    targets: np.ndarray


def make_synthetic_batch(
    kind: str,
    batch_size: int,
    channels: int,
    height: int,
    width: int,
    seed: int,
) -> SampleBatch:
    """Build a deterministic synthetic batch with a planted rectangular blob.

    Args:
        kind: ``classifier`` or ``segmentor``.
        batch_size: Number of samples N.
        channels: Band count C.
        height: Spatial height H.
        width: Spatial width W.
        seed: RNG seed.

    Returns:
        SampleBatch: Images in [0, 1]. Classifier targets are 1.0 when a blob
        is planted (even indices). Segmentor targets are the blob mask.

    Raises:
        ValueError: If kind is not classifier or segmentor.
    """
    rng = np.random.default_rng(seed)
    images = rng.uniform(0.05, 0.15, size=(batch_size, channels, height, width)).astype(np.float32)
    y0 = height // 4
    y1 = (3 * height) // 4
    x0 = width // 4
    x1 = (3 * width) // 4
    if kind == "segmentor":
        masks = np.zeros((batch_size, 1, height, width), dtype=np.float32)
        masks[:, 0, y0:y1, x0:x1] = 1.0
        images[:, :, y0:y1, x0:x1] = np.clip(images[:, :, y0:y1, x0:x1] + 0.7, 0.0, 1.0)
        return SampleBatch(images=images, targets=masks)
    if kind == "classifier":
        labels = np.zeros((batch_size, 1), dtype=np.float32)
        for i in range(batch_size):
            if i % 2 == 0:
                labels[i, 0] = 1.0
                images[i, :, y0:y1, x0:x1] = np.clip(images[i, :, y0:y1, x0:x1] + 0.7, 0.0, 1.0)
        return SampleBatch(images=images, targets=labels)
    raise ValueError(f"unknown train kind {kind!r}")


def _maybe_normalize(images: np.ndarray, bit_depth: int) -> np.ndarray:
    """Scale DN-valued planes through `normalize_dn`; leave [0, 1] tensors.

    Args:
        images: np.ndarray[float32, (N, C, H, W)] or a single (C, H, W).
        bit_depth: ADC bit depth used when values exceed 1.0.

    Returns:
        np.ndarray[float32] in [0, 1] with the same rank as `images`.
    """
    arr = np.asarray(images, dtype=np.float32)
    if float(np.nanmax(arr)) <= 1.0:
        return np.clip(arr, 0.0, 1.0)
    if arr.ndim == 3:
        return normalize_dn(arr, bit_depth)
    planes = [normalize_dn(arr[i], bit_depth) for i in range(arr.shape[0])]
    return np.stack(planes, axis=0)


def load_disk_batch(
    data_dir: str | Path,
    kind: str,
    bit_depth: int = 12,
) -> SampleBatch:
    """Load a packed numpy batch from `data_dir`.

    Args:
        data_dir: Directory with `images.npy` and either `labels.npy`
            (classifier) or `masks.npy` (segmentor).
        kind: ``classifier`` or ``segmentor``.
        bit_depth: ADC bit depth for DN-valued `images.npy`.

    Returns:
        SampleBatch: Images in [0, 1]. Targets match `kind`.

    Raises:
        FileNotFoundError: If a required npy file is missing.
        ValueError: If kind is unknown or array ranks do not match.
    """
    root = Path(data_dir)
    images = _maybe_normalize(np.load(root / "images.npy"), bit_depth)
    if images.ndim != 4:
        raise ValueError(f"images.npy must have shape (N, C, H, W); got {images.shape}")
    if kind == "classifier":
        labels = np.asarray(np.load(root / "labels.npy"), dtype=np.float32)
        if labels.ndim == 1:
            labels = labels.reshape(-1, 1)
        if labels.shape[0] != images.shape[0]:
            raise ValueError("labels.npy N does not match images.npy N")
        return SampleBatch(images=images, targets=labels)
    if kind == "segmentor":
        masks = np.asarray(np.load(root / "masks.npy"), dtype=np.float32)
        if masks.ndim == 3:
            masks = masks[:, np.newaxis, ...]
        if masks.shape[0] != images.shape[0]:
            raise ValueError("masks.npy N does not match images.npy N")
        if masks.shape[-2:] != images.shape[-2:]:
            raise ValueError("masks.npy spatial size does not match images.npy")
        return SampleBatch(images=images, targets=masks)
    raise ValueError(f"unknown train kind {kind!r}")
