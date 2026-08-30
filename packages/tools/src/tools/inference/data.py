"""Training tensors: synthetic scenes and an on-disk pack adapter.

Images are float32 in the flight `normalize_dn` domain unless the disk loader
sees DN-valued arrays and rescales them. In-memory batches are torch tensors.
On-disk packs stay `images.npy`, `masks.npy`, `labels.npy`, `splits.json`, and
`dataset.json`.

Contains:
  - SampleBatch: one image tensor plus classifier or segmentor targets.
  - ProcessedPack: memmap or tensor images plus split index and dataset hash.
  - _row_image: copy one pack image row to a CPU tensor.
  - SplitDataset: torch Dataset over one named split.
  - apply_train_augment: seeded flip and 90-degree rotation on the train split.
  - make_synthetic_batch: planted-blob scenes for a 1-step train smoke test.
  - make_synthetic_pack: even-index blobs with masks and labels.
  - write_processed_pack: persist tensors, splits, and dataset.json.
  - load_disk_batch: read `images.npy` plus `labels.npy` or `masks.npy`.
  - load_processed_pack / load_split: split-aware pack loader. Classifier
    loads skip ``masks.npy``.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from flight.payload.preprocess import normalize_dn
from torch.utils.data import Dataset

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


def _as_numpy(array: torch.Tensor | np.ndarray) -> np.ndarray:
    """Return a contiguous float32 ndarray for `.npy` I/O.

    Args:
        array: Torch tensor or numpy array.

    Returns:
        np.ndarray[float32]: CPU contiguous copy when needed.
    """
    if isinstance(array, torch.Tensor):
        return np.ascontiguousarray(array.detach().cpu().numpy(), dtype=np.float32)
    return np.ascontiguousarray(array, dtype=np.float32)


def _as_tensor(array: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Return a CPU float32 tensor, sharing memory when the ndarray is writable.

    Args:
        array: Torch tensor or numpy array.

    Returns:
        torch.Tensor: float32 tensor on CPU.
    """
    if isinstance(array, torch.Tensor):
        return array.detach().to(dtype=torch.float32, device="cpu")
    arr = np.ascontiguousarray(array, dtype=np.float32)
    if not arr.flags.writeable:
        arr = np.array(arr, dtype=np.float32, copy=True)
    return torch.from_numpy(arr)


@dataclass(frozen=True, slots=True)
class SampleBatch:
    """One training batch in torch.

    Attributes:
        images: torch.Tensor[float32, (N, C, H, W)] in [0, 1] on CPU.
        targets: Classifier (N, 1) float32 labels, or segmentor (N, 1, H, W)
            float32 masks in {0, 1}.
    """

    images: torch.Tensor
    targets: torch.Tensor


_MATERIALIZE_BYTES = 512 * 1024 * 1024


def _row_image(store: torch.Tensor | np.ndarray, index: int) -> torch.Tensor:
    """Return one ``(C, H, W)`` float32 CPU image, copying a memmap row.

    Args:
        store: Full pack image array or tensor.
        index: Pack index.

    Returns:
        torch.Tensor: One sample. A numpy row is copied so the caller owns it.
    """
    item = store[index]
    if isinstance(item, torch.Tensor):
        return item
    row = np.array(item, dtype=np.float32, copy=True)
    return torch.from_numpy(np.ascontiguousarray(row))


@dataclass(frozen=True, slots=True)
class ProcessedPack:
    """On-disk processed corpus with frozen splits.

    Attributes:
        images: ``(N, C, H, W)`` float32 in ``[0, 1]``. Small packs are a CPU
            tensor. Packs whose ``images.npy`` exceeds ``_MATERIALIZE_BYTES``
            keep a read-only memmap so the loader does not commit the file.
        masks: torch.Tensor[float32, (N, 1, H, W)] in {0, 1} on CPU. When
            masks were not requested, this is a ``(N, 1, 1, 1)`` zero tensor.
        labels: torch.Tensor[float32, (N, 1)] in {0, 1} on CPU.
        splits: Frozen train/val/test indices.
        meta: Pack identity including dataset_hash.
        pack_dir: Filesystem directory of the pack.
    """

    images: torch.Tensor | np.ndarray
    masks: torch.Tensor
    labels: torch.Tensor
    splits: SplitIndex
    meta: DatasetMeta
    pack_dir: Path


class SplitDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """One named split of a processed pack as a torch Dataset.

    Each item is a single sample: image ``(C, H, W)`` and a classifier
    ``(1,)`` label or a segmentor ``(1, H, W)`` mask.
    """

    def __init__(
        self,
        pack: ProcessedPack,
        kind: str,
        split: str,
        augment: bool = False,
        seed: int = 0,
    ) -> None:
        """Index one split of ``pack`` for ``kind``.

        Args:
            pack: Loaded processed pack.
            kind: ``classifier`` or ``segmentor``.
            split: ``train``, ``val``, or ``test``.
            augment: When true, apply train-only flip and rotation.
            seed: RNG seed combined with the sample index for aug.

        Raises:
            ValueError: If kind is unknown or the split is empty.
        """
        if kind not in {"classifier", "segmentor"}:
            raise ValueError(f"unknown train kind {kind!r}")
        indices = pack.splits.for_name(split)
        if not indices:
            raise ValueError(f"split {split!r} is empty")
        self._pack = pack
        self._kind = kind
        self._indices = indices
        self._augment = bool(augment)
        self._seed = int(seed)
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Advance the augmentation stream to a new epoch.

        Args:
            epoch: Epoch number, counted from one by the train loop.

        Notes:
            The transform for a sample is drawn from ``(seed, epoch, index)``.
            Without the epoch term every sample would keep one fixed flip and
            rotation for the whole run, which permutes the dataset once rather
            than augmenting it.
        """
        self._epoch = int(epoch)

    def __len__(self) -> int:
        """Return the number of samples in the split."""
        return len(self._indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(image, target)`` for one split position.

        Args:
            index: Position in the split, not the pack index.

        Returns:
            tuple: Image ``(C, H, W)`` and target matching ``kind``.
        """
        idx = self._indices[int(index)]
        image = _row_image(self._pack.images, idx)
        if self._kind == "classifier":
            target = self._pack.labels[idx]
            if target.ndim == 0:
                target = target.reshape(1)
        else:
            target = self._pack.masks[idx]
            if target.ndim == 2:
                target = target.unsqueeze(0)
        if self._augment:
            image, target = apply_train_augment(
                image, target, self._kind, self._seed, int(index), self._epoch
            )
        return image, target


def apply_train_augment(
    image: torch.Tensor,
    target: torch.Tensor,
    kind: str,
    seed: int,
    index: int,
    epoch: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a seeded horizontal flip and 90-degree rotation.

    Args:
        image: torch.Tensor[float32, (C, H, W)].
        target: Classifier ``(1,)`` label or segmentor ``(1, H, W)`` mask.
        kind: ``classifier`` or ``segmentor``.
        seed: Train seed.
        index: Dataset index in the split.
        epoch: Epoch number. Samples take a new transform each epoch, which is
            what makes this augmentation rather than a one-time permutation.

    Returns:
        tuple: Transformed image and target. Classifier labels stay unchanged.
    """
    rng = torch.Generator()
    rng.manual_seed(int(seed) + int(index) * 1_000_003 + int(epoch) * 7_919)
    out_image = image
    out_target = target
    if float(torch.rand(1, generator=rng).item()) >= 0.5:
        out_image = torch.flip(out_image, dims=(-1,))
        if kind == "segmentor":
            out_target = torch.flip(out_target, dims=(-1,))
    turns = int(torch.randint(0, 4, (1,), generator=rng).item())
    if turns:
        out_image = torch.rot90(out_image, turns, dims=(-2, -1))
        if kind == "segmentor":
            out_target = torch.rot90(out_target, turns, dims=(-2, -1))
    return out_image, out_target


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
    generator = torch.Generator()
    generator.manual_seed(seed)
    images = 0.05 + 0.10 * torch.rand(
        (batch_size, channels, height, width), generator=generator, dtype=torch.float32
    )
    y0 = height // 4
    y1 = (3 * height) // 4
    x0 = width // 4
    x1 = (3 * width) // 4
    if kind == "segmentor":
        masks = torch.zeros((batch_size, 1, height, width), dtype=torch.float32)
        masks[:, 0, y0:y1, x0:x1] = 1.0
        images[:, :, y0:y1, x0:x1] = torch.clamp(images[:, :, y0:y1, x0:x1] + 0.7, 0.0, 1.0)
        return SampleBatch(images=images, targets=masks)
    if kind == "classifier":
        labels = torch.zeros((batch_size, 1), dtype=torch.float32)
        for i in range(batch_size):
            if i % 2 == 0:
                labels[i, 0] = 1.0
                images[i, :, y0:y1, x0:x1] = torch.clamp(images[i, :, y0:y1, x0:x1] + 0.7, 0.0, 1.0)
        return SampleBatch(images=images, targets=labels)
    raise ValueError(f"unknown train kind {kind!r}")


def make_synthetic_pack(
    n: int,
    channels: int,
    height: int,
    width: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    generator = torch.Generator()
    generator.manual_seed(seed)
    images = 0.05 + 0.10 * torch.rand(
        (n, channels, height, width), generator=generator, dtype=torch.float32
    )
    masks = torch.zeros((n, 1, height, width), dtype=torch.float32)
    y0 = height // 4
    y1 = (3 * height) // 4
    x0 = width // 4
    x1 = (3 * width) // 4
    for i in range(n):
        if i % 2 == 0:
            masks[i, 0, y0:y1, x0:x1] = 1.0
            images[i, :, y0:y1, x0:x1] = torch.clamp(images[i, :, y0:y1, x0:x1] + 0.7, 0.0, 1.0)
    labels = (masks.reshape(n, -1).amax(dim=1) > 0.0).to(dtype=torch.float32).reshape(n, 1)
    return images, masks, labels


def write_processed_pack(
    dest_dir: str | Path,
    images: torch.Tensor | np.ndarray,
    masks: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
    recipe: SplitRecipe,
    source_doi: str = "",
) -> DatasetMeta:
    """Write a processed pack with splits.json and dataset.json.

    Args:
        dest_dir: Output directory.
        images: torch.Tensor or np.ndarray[float32, (N, C, H, W)].
        masks: torch.Tensor or np.ndarray[float32, (N, 1, H, W)].
        labels: torch.Tensor or np.ndarray[float32, (N, 1)].
        recipe: Split recipe applied to ``range(N)``.
        source_doi: DOI recorded in dataset.json. Empty for synthetic packs.

    Returns:
        DatasetMeta: Written identity including dataset_hash.

    Raises:
        ValueError: If ranks, N, or spatial sizes do not match.
    """
    img = _as_numpy(images)
    mask = _as_numpy(masks)
    lab = _as_numpy(labels)
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
    np.save(root / "images.npy", img)
    np.save(root / "masks.npy", mask)
    np.save(root / "labels.npy", lab)
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
    """Load a packed batch from `data_dir`.

    Args:
        data_dir: Directory with `images.npy` and either `labels.npy`
            (classifier) or `masks.npy` (segmentor).
        kind: ``classifier`` or ``segmentor``.
        bit_depth: ADC bit depth for DN-valued `images.npy`.

    Returns:
        SampleBatch: Images in [0, 1] as CPU float32 tensors. Targets match
        `kind`.

    Raises:
        FileNotFoundError: If a required npy file is missing.
        ValueError: If kind is unknown or array ranks do not match.
    """
    root = Path(data_dir)
    images = _as_tensor(_maybe_normalize(np.load(root / "images.npy"), bit_depth))
    if images.ndim != 4:
        raise ValueError(f"images.npy must have shape (N, C, H, W); got {tuple(images.shape)}")
    if kind == "classifier":
        labels = _as_tensor(np.load(root / "labels.npy"))
        if labels.ndim == 1:
            labels = labels.reshape(-1, 1)
        if int(labels.shape[0]) != int(images.shape[0]):
            raise ValueError("labels.npy N does not match images.npy N")
        return SampleBatch(images=images, targets=labels)
    if kind == "segmentor":
        masks = _as_tensor(np.load(root / "masks.npy"))
        if masks.ndim == 3:
            masks = masks.unsqueeze(1)
        if int(masks.shape[0]) != int(images.shape[0]):
            raise ValueError("masks.npy N does not match images.npy N")
        if tuple(masks.shape[-2:]) != tuple(images.shape[-2:]):
            raise ValueError("masks.npy spatial size does not match images.npy")
        return SampleBatch(images=images, targets=masks)
    raise ValueError(f"unknown train kind {kind!r}")


def load_processed_pack(
    data_dir: str | Path,
    bit_depth: int = 12,
    load_masks: bool = True,
) -> ProcessedPack:
    """Load a processed pack with memmap-backed tensors and frozen splits.

    Args:
        data_dir: Directory with the pack files.
        bit_depth: ADC bit depth for DN-valued `images.npy`.
        load_masks: When false, skip ``masks.npy`` and store a ``(N, 1, 1, 1)``
            zero tensor. Classifier loaders do not read masks.

    Returns:
        ProcessedPack: CPU tensors or an image memmap, splits, and meta.

    Raises:
        FileNotFoundError: If a required pack file is missing.
        ValueError: If dataset.json hash does not match the files, or ranks
            do not match.

    Notes:
        Probe one sample to choose the DN versus unit-interval branch. A pack
        already in ``[0, 1]`` whose ``images.npy`` exceeds
        ``_MATERIALIZE_BYTES`` stays a read-only memmap. Copy-on-write of a
        multi-gigabyte file exhausts the Windows page file.
    """
    root = Path(data_dir)
    raw_images = np.load(root / "images.npy", mmap_mode="r")
    if raw_images.ndim != 4:
        raise ValueError(f"images.npy must have shape (N, C, H, W); got {raw_images.shape}")
    probe = np.asarray(raw_images[0], dtype=np.float32)
    unit_interval = float(np.nanmax(probe)) <= 1.0
    images: torch.Tensor | np.ndarray
    if unit_interval and int(raw_images.nbytes) > _MATERIALIZE_BYTES:
        images = raw_images
    elif unit_interval:
        images = _as_tensor(raw_images)
    else:
        images = _as_tensor(_maybe_normalize(np.asarray(raw_images), bit_depth))
    n = int(raw_images.shape[0])
    if load_masks:
        masks_np = np.load(root / "masks.npy", mmap_mode="r")
        if masks_np.ndim == 3:
            masks_np = np.asarray(masks_np, dtype=np.float32)[:, np.newaxis, ...]
        masks = _as_tensor(masks_np)
        if int(masks.shape[0]) != n:
            raise ValueError("masks.npy N does not match images.npy N")
    else:
        masks = torch.zeros((n, 1, 1, 1), dtype=torch.float32)
    labels_np = np.load(root / "labels.npy", mmap_mode="r")
    if labels_np.ndim == 1:
        labels_np = np.asarray(labels_np, dtype=np.float32).reshape(-1, 1)
    labels = _as_tensor(labels_np)
    if int(labels.shape[0]) != n:
        raise ValueError("labels.npy N does not match images.npy N")
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
            height=int(raw_images.shape[2]),
            width=int(raw_images.shape[3]),
            in_channels=int(raw_images.shape[1]),
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
) -> SplitDataset:
    """Load one named split from a processed pack as a Dataset.

    Args:
        data_dir: Pack directory.
        kind: ``classifier`` or ``segmentor``.
        split: ``train``, ``val``, or ``test``.
        bit_depth: ADC bit depth for DN-valued images.

    Returns:
        SplitDataset: One sample per index in the named split.

    Raises:
        ValueError: If kind or split is unknown, or the split is empty.
        FileNotFoundError: If pack files are missing.
    """
    pack = load_processed_pack(data_dir, bit_depth=bit_depth, load_masks=kind != "classifier")
    return SplitDataset(pack, kind, split)
