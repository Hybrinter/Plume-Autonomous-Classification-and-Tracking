"""Torch datasets over finished shards.

Contains:
  - ShardDataset: one ``<task>/<split>/<H>x<W>`` directory.
  - make_loader: seeded batches that each come from a single shard.

This is the only module in ``tools.ml_models.dataset`` that imports torch.
A batch is ``(image, g, target)``. ``image`` is float32 unit ``(B, C, H, W)``.
``g`` is float32 ``(B, 2)`` from ``to_model_gsd``. The classifier target is
``(B, 1)``. The segmentor target is ``(B, 1, H, W)``.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from tools.ml_models.dataset.gsd import to_model_gsd
from tools.ml_models.dataset.manifest import check_compatible, load_manifest, parse_shard_size

Batch = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class ShardDataset(Dataset[Batch]):
    """One finished shard.

    Attributes:
        shard_dir: Directory that holds the ``.npy`` files and ``rows.jsonl``.
    """

    def __init__(
        self, shard_dir: str | Path, gsd_reference_m: float, task: str, *, channels: int
    ) -> None:
        """Memory-map a shard.

        Args:
            shard_dir: Shard directory.
            gsd_reference_m: Reference passed to ``to_model_gsd``.
            task: ``classifier`` or ``segmentor``.
            channels: Expected image channel count.

        Raises:
            ValueError: If ``task`` is unknown, the stored arrays are not the
                expected dtype and layout, or a segmentor shard has no masks.
            FileNotFoundError: If an array file is missing.
        """
        if task not in ("classifier", "segmentor"):
            raise ValueError(f"unknown task {task!r}")
        if channels < 1:
            raise ValueError(f"channels must be >= 1; got {channels}")
        self.shard_dir = Path(shard_dir)
        self._reference_m = gsd_reference_m
        self._task = task
        self._images = np.load(self.shard_dir / "images.npy", mmap_mode="r")
        if (
            self._images.dtype != np.float32
            or self._images.ndim != 4
            or self._images.shape[1] != channels
        ):
            raise ValueError(
                f"images.npy must be float32 (N, {channels}, H, W); "
                f"got {self._images.dtype} {self._images.shape}"
            )
        count, _, height, width = self._images.shape
        if count < 1 or height < 1 or width < 1:
            raise ValueError(f"images.npy dims must be positive; got {self._images.shape}")
        self._gsd = np.load(self.shard_dir / "gsd.npy", mmap_mode="r")
        if self._gsd.dtype != np.float32 or self._gsd.shape != (count, 2):
            raise ValueError(
                f"gsd.npy must be float32 ({count}, 2); got {self._gsd.dtype} {self._gsd.shape}"
            )
        self._labels = np.load(self.shard_dir / "labels.npy", mmap_mode="r")
        if self._labels.dtype != np.float32 or self._labels.shape != (count, 1):
            raise ValueError(
                f"labels.npy must be float32 ({count}, 1); "
                f"got {self._labels.dtype} {self._labels.shape}"
            )
        mask_path = self.shard_dir / "masks.npy"
        if task == "segmentor" and not mask_path.is_file():
            raise ValueError(f"segmentor shard {self.shard_dir} has no masks.npy")
        self._masks = np.load(mask_path, mmap_mode="r") if mask_path.is_file() else None
        if self._masks is not None and (
            self._masks.dtype != np.uint8 or self._masks.shape != (count, 1, height, width)
        ):
            raise ValueError(
                f"masks.npy must be uint8 ({count}, 1, {height}, {width}); "
                f"got {self._masks.dtype} {self._masks.shape}"
            )

    def __len__(self) -> int:
        """Return the row count.

        Returns:
            int: Leading dimension of ``images.npy``.
        """
        return int(self._images.shape[0])

    def __getitem__(self, index: int) -> Batch:
        """Return one row as unit image, model GSD, and target.

        Args:
            index: Row index.

        Returns:
            Batch: ``(image float32 (C, H, W), g float32 (2,), target)``.

        Raises:
            ValueError: If the stored row is not finite inside ``[0, 1]``.
        """
        stored = np.array(self._images[index], dtype=np.float32, copy=True, order="C")
        if not np.all(np.isfinite(stored)) or not np.all((stored >= 0.0) & (stored <= 1.0)):
            raise ValueError(f"stored image {index} in {self.shard_dir} is not unit float32")
        image = torch.from_numpy(stored)
        encoded = to_model_gsd(np.asarray(self._gsd[index], dtype=np.float32), self._reference_m)
        gsd = torch.from_numpy(np.ascontiguousarray(encoded))
        if self._task == "segmentor":
            if self._masks is None:
                raise ValueError(f"segmentor shard {self.shard_dir} has no masks.npy")
            target = torch.from_numpy(np.array(self._masks[index], dtype=np.float32, copy=True))
        else:
            target = torch.from_numpy(np.array(self._labels[index], dtype=np.float32, copy=True))
        return image, gsd, target


def make_loader(
    dests: Sequence[str | Path],
    task: str,
    split: str,
    batch_size: int,
    weights: Sequence[float] | None,
    seed: int,
    *,
    n_batches: int,
) -> Iterator[Batch]:
    """Yield seeded single-shard batches.

    Args:
        dests: Finished dataset directories.
        task: ``classifier`` or ``segmentor``.
        split: ``train``, ``val``, or ``test``.
        batch_size: Rows per batch. A shard smaller than this is sampled with
            replacement.
        weights: Relative dataset weights. None uses equal weights.
        seed: NumPy Generator seed.
        n_batches: Number of batches to yield.

    Returns:
        Iterator[Batch]: Each batch is a ``(image, g, target)`` triple collated
        from one shard by a single-batch ``DataLoader``. The dataset is drawn
        with probability proportional to ``weights``. The shard inside that
        dataset is drawn with probability proportional to its row count.

    Raises:
        ValueError: If the datasets are incompatible, a weight is not positive,
            or a dataset has no shard for the requested task and split.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1; got {batch_size}")
    if n_batches < 1:
        raise ValueError(f"n_batches must be >= 1; got {n_batches}")
    if len(dests) < 1:
        raise ValueError("make_loader requires at least one dataset")
    manifests = [load_manifest(Path(dest) / "dataset.json") for dest in dests]
    check_compatible(manifests)
    grouped: list[list[ShardDataset]] = []
    for dest, manifest in zip(dests, manifests, strict=True):
        parent = Path(dest) / task / split
        shards: list[ShardDataset] = []
        if parent.is_dir():
            for child in sorted(path for path in parent.iterdir() if path.is_dir()):
                if parse_shard_size(child.name) is None:
                    continue
                shards.append(
                    ShardDataset(
                        child,
                        manifest.gsd_reference_m,
                        task,
                        channels=len(manifest.band_names),
                    )
                )
        if not shards:
            raise ValueError(f"no {task}/{split} shards in {dest}")
        grouped.append(shards)
    probs = _dataset_probs(weights, len(dests))
    generator = np.random.default_rng(seed)
    for _ in range(n_batches):
        dataset_index = int(generator.choice(len(grouped), p=probs))
        shards = grouped[dataset_index]
        counts = np.asarray([len(shard) for shard in shards], dtype=np.float64)
        shard_index = int(generator.choice(len(shards), p=counts / counts.sum()))
        shard = shards[shard_index]
        replace = batch_size > len(shard)
        chosen = generator.choice(len(shard), size=batch_size, replace=replace)
        loader = DataLoader(
            Subset(shard, [int(row) for row in chosen]),
            batch_size=len(chosen),
            shuffle=False,
        )
        yield next(iter(loader))


def _dataset_probs(weights: Sequence[float] | None, count: int) -> np.ndarray:
    """Normalize dataset sampling weights.

    Args:
        weights: Relative weights, or None for equal weight.
        count: Number of datasets.

    Returns:
        np.ndarray[float64, (count,)]: Probabilities that sum to 1.

    Raises:
        ValueError: If the length disagrees or a weight is not finite and > 0.
    """
    if weights is None:
        values = np.ones(count, dtype=np.float64)
    else:
        if len(weights) != count:
            raise ValueError(f"weights length {len(weights)} != dataset count {count}")
        values = np.asarray(weights, dtype=np.float64)
    if values.shape != (count,) or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("weights must be finite and > 0")
    return np.asarray(values / values.sum())
