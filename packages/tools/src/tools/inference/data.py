"""Training tensors: synthetic scenes and an on-disk numpy adapter.

This module stays torch-free. Images are float32 in the flight `normalize_dn`
domain unless the disk loader sees DN-valued arrays and rescales them.

A processed pack is `images.npy` plus `masks.npy`, `labels.npy`, `splits.json`,
and `dataset.json`. `load_split` indexes one named split through a memmap.

Contains:
  - SampleBatch: one image tensor plus classifier or segmentor targets.
  - ProcessedPack: memmap views plus split index and dataset hash.
  - make_synthetic_batch: planted-blob scenes for a 1-step train smoke test.
  - make_synthetic_pack: even-index blobs with masks and labels.
  - write_processed_pack: persist tensors, splits, and dataset.json.
  - load_disk_batch: read `images.npy` plus `labels.npy` or `masks.npy`.
  - load_processed_pack / load_split: split-aware pack loader.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from flight.payload.preprocess import normalize_dn

from tools.inference.split import (
    DatasetMeta,
    SplitIndex,
    SplitRecipe,
    assign_splits,
    compute_dataset_hash,
    load_dataset_meta,
    load_splits,
    write_dataset_meta,
    write_splits,
)


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


@dataclass(frozen=True, slots=True)
class ProcessedPack:
    """On-disk processed corpus with frozen splits.

    Attributes:
        images: np.ndarray[float32, (N, C, H, W)] memmap or array in [0, 1].
        masks: np.ndarray[float32, (N, 1, H, W)] memmap or array in {0, 1}.
        labels: np.ndarray[float32, (N, 1)] memmap or array in {0, 1}.
        splits: Frozen train/val/test indices.
        meta: Pack identity including dataset_hash.
        pack_dir: Filesystem directory of the pack.
    """

    images: np.ndarray
    masks: np.ndarray
    labels: np.ndarray
    splits: SplitIndex
    meta: DatasetMeta
    pack_dir: Path


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


def make_synthetic_pack(
    n: int,
    channels: int,
    height: int,
    width: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a labeled synthetic pack with blobs on even indices.

    Args:
        n: Sample count N.
        channels: Band count C.
        height: Spatial height H.
        width: Spatial width W.
        seed: RNG seed.

    Returns:
        tuple: ``(images, masks, labels)``. Images are (N, C, H, W) float32 in
        [0, 1]. Masks are (N, 1, H, W). Labels are (N, 1). Even indices are
        positive.

    Notes:
        One pack serves both classifier and segmentor loaders. Labels are 1.0
        iff the mask has any positive pixel.
    """
    rng = np.random.default_rng(seed)
    images = rng.uniform(0.05, 0.15, size=(n, channels, height, width)).astype(np.float32)
    masks = np.zeros((n, 1, height, width), dtype=np.float32)
    y0 = height // 4
    y1 = (3 * height) // 4
    x0 = width // 4
    x1 = (3 * width) // 4
    for i in range(n):
        if i % 2 == 0:
            masks[i, 0, y0:y1, x0:x1] = 1.0
            images[i, :, y0:y1, x0:x1] = np.clip(images[i, :, y0:y1, x0:x1] + 0.7, 0.0, 1.0)
    labels = (masks.reshape(n, -1).max(axis=1) > 0.0).astype(np.float32).reshape(n, 1)
    return images, masks, labels


def write_processed_pack(
    dest_dir: str | Path,
    images: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    recipe: SplitRecipe,
    source_doi: str = "",
) -> DatasetMeta:
    """Write a processed pack with splits.json and dataset.json.

    Args:
        dest_dir: Output directory.
        images: np.ndarray[float32, (N, C, H, W)].
        masks: np.ndarray[float32, (N, 1, H, W)].
        labels: np.ndarray[float32, (N, 1)].
        recipe: Split recipe applied to ``range(N)``.
        source_doi: DOI recorded in dataset.json. Empty for synthetic packs.

    Returns:
        DatasetMeta: Written identity including dataset_hash.

    Raises:
        ValueError: If ranks, N, or spatial sizes do not match.
    """
    img = np.asarray(images, dtype=np.float32)
    mask = np.asarray(masks, dtype=np.float32)
    lab = np.asarray(labels, dtype=np.float32)
    if img.ndim != 4:
        raise ValueError(f"images must have shape (N, C, H, W); got {img.shape}")
    if mask.ndim == 3:
        mask = mask[:, np.newaxis, ...]
    if lab.ndim == 1:
        lab = lab.reshape(-1, 1)
    n = int(img.shape[0])
    if mask.shape[0] != n or lab.shape[0] != n:
        raise ValueError("masks and labels N must match images N")
    if mask.shape[-2:] != img.shape[-2:]:
        raise ValueError("masks spatial size must match images")
    root = Path(dest_dir)
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "images.npy", np.ascontiguousarray(img, dtype=np.float32))
    np.save(root / "masks.npy", np.ascontiguousarray(mask, dtype=np.float32))
    np.save(root / "labels.npy", np.ascontiguousarray(lab, dtype=np.float32))
    write_splits(root / "splits.json", assign_splits(n, recipe))
    digest = compute_dataset_hash(root)
    meta = DatasetMeta(
        dataset_hash=digest,
        source_doi=source_doi,
        n=n,
        height=int(img.shape[2]),
        width=int(img.shape[3]),
        in_channels=int(img.shape[1]),
    )
    write_dataset_meta(root / "dataset.json", meta)
    return meta


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


def load_processed_pack(data_dir: str | Path, bit_depth: int = 12) -> ProcessedPack:
    """Load a processed pack with memmap tensors and frozen splits.

    Args:
        data_dir: Directory with the pack files.
        bit_depth: ADC bit depth for DN-valued `images.npy`.

    Returns:
        ProcessedPack: Memmap views, splits, and meta.

    Raises:
        FileNotFoundError: If a required pack file is missing.
        ValueError: If dataset.json hash does not match the files, or ranks
            do not match.

    Notes:
        Images load through a memmap. DN-valued images copy into a normalized
        array. Masks and labels stay memmap views.
    """
    root = Path(data_dir)
    raw_images = np.load(root / "images.npy", mmap_mode="r")
    if raw_images.ndim != 4:
        raise ValueError(f"images.npy must have shape (N, C, H, W); got {raw_images.shape}")
    if float(np.nanmax(raw_images)) <= 1.0:
        images = raw_images
    else:
        images = _maybe_normalize(np.asarray(raw_images), bit_depth)
    masks = np.load(root / "masks.npy", mmap_mode="r")
    if masks.ndim == 3:
        masks = np.asarray(masks, dtype=np.float32)[:, np.newaxis, ...]
    labels = np.load(root / "labels.npy", mmap_mode="r")
    if labels.ndim == 1:
        labels = np.asarray(labels, dtype=np.float32).reshape(-1, 1)
    n = int(images.shape[0])
    if int(masks.shape[0]) != n or int(labels.shape[0]) != n:
        raise ValueError("masks.npy or labels.npy N does not match images.npy N")
    splits = load_splits(root / "splits.json")
    digest = compute_dataset_hash(root)
    meta_path = root / "dataset.json"
    if meta_path.is_file():
        meta = load_dataset_meta(meta_path)
        if meta.dataset_hash != digest:
            raise ValueError("dataset.json hash does not match pack files")
    else:
        meta = DatasetMeta(
            dataset_hash=digest,
            source_doi="",
            n=n,
            height=int(images.shape[2]),
            width=int(images.shape[3]),
            in_channels=int(images.shape[1]),
        )
    return ProcessedPack(
        images=images,
        masks=masks,
        labels=labels,
        splits=splits,
        meta=meta,
        pack_dir=root,
    )


def load_split(
    data_dir: str | Path,
    kind: str,
    split: str,
    bit_depth: int = 12,
) -> SampleBatch:
    """Load one named split from a processed pack.

    Args:
        data_dir: Pack directory.
        kind: ``classifier`` or ``segmentor``.
        split: ``train``, ``val``, or ``test``.
        bit_depth: ADC bit depth for DN-valued images.

    Returns:
        SampleBatch: Contiguous float32 arrays for the selected indices.

    Raises:
        ValueError: If kind or split is unknown, or the split is empty.
        FileNotFoundError: If pack files are missing.
    """
    pack = load_processed_pack(data_dir, bit_depth=bit_depth)
    indices = pack.splits.for_name(split)
    if not indices:
        raise ValueError(f"split {split!r} is empty")
    idx = np.asarray(indices, dtype=np.int64)
    images = np.ascontiguousarray(pack.images[idx], dtype=np.float32)
    if kind == "classifier":
        targets = np.ascontiguousarray(pack.labels[idx], dtype=np.float32)
        if targets.ndim == 1:
            targets = targets.reshape(-1, 1)
        return SampleBatch(images=images, targets=targets)
    if kind == "segmentor":
        targets = np.ascontiguousarray(pack.masks[idx], dtype=np.float32)
        if targets.ndim == 3:
            targets = targets[:, np.newaxis, ...]
        return SampleBatch(images=images, targets=targets)
    raise ValueError(f"unknown train kind {kind!r}")
