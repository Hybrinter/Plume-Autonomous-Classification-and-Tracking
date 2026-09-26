"""On-disk processed packs and in-memory concatenation.

Contains:
  - ProcessedPack: arrays, split index, and dataset meta.
  - write_processed_pack / load_processed_pack: ``images.npy``, ``masks.npy``,
    ``labels.npy``, ``splits.json``, ``provenance.json``, and ``dataset.json``.
  - concat_packs / assert_same_ingest: stack packs that share one provenance.

``write_processed_pack`` writes provenance, then the pack hash, then
``dataset.json``. ``concat_packs`` does not write files. Its ``dataset_hash``
is empty.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tools.ml_models.data.meta import (
    DatasetMeta,
    Provenance,
    compute_dataset_hash,
    dataset_meta_from_provenance,
    load_dataset_meta,
    load_provenance,
    provenance_from_meta,
    write_dataset_meta,
    write_provenance,
)
from tools.ml_models.data.split import (
    SplitIndex,
    SplitRecipe,
    assign_group_splits,
    load_splits,
    write_splits,
)


@dataclass(frozen=True, slots=True)
class ProcessedPack:
    """In-memory processed pack.

    Attributes:
        images: np.ndarray[float32, (N, C, H, W)], or a read-only memmap of that shape.
        masks: np.ndarray[float32, (N, 1, H, W)].
        labels: np.ndarray[float32, (N, 1)].
        splits: Row indices for train, val, and test.
        meta: Pack identity. ``dataset_hash`` is empty on a concatenated pack.
    """

    images: np.ndarray
    masks: np.ndarray
    labels: np.ndarray
    splits: SplitIndex
    meta: DatasetMeta


def write_processed_pack(
    dest: str | Path,
    images: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    provenance: Provenance,
    source_doi: str,
    *,
    group_ids: Sequence[str] | None = None,
    splits: SplitIndex | None = None,
    recipe: SplitRecipe | None = None,
) -> DatasetMeta:
    """Write a processed pack and return its dataset sidecar.

    Args:
        dest: Output directory.
        images: np.ndarray[float32, (N, C, H, W)]. ``C`` equals
            ``len(provenance.band_names)``.
        masks: np.ndarray[float32, (N, 1, H, W)].
        labels: np.ndarray[float32, (N, 1)].
        provenance: Ingest and normalization record stored on the meta.
        source_doi: DOI recorded in ``dataset.json``.
        group_ids: Group id per row. Required when ``splits`` is omitted.
            Length must equal N. Passed to ``assign_group_splits``.
        splits: Precomputed index. Required when ``group_ids`` is omitted.
        recipe: Fractions for ``assign_group_splits``. Defaults to 0.70/0.15/0.15.
            Used only with ``group_ids``.

    Returns:
        DatasetMeta: Written identity, including ``dataset_hash``.

    Raises:
        ValueError: If shapes, dtypes, or the split source are invalid, or both
            ``group_ids`` and ``splits`` are passed.

    Notes:
        Files are written as arrays, ``splits.json``, then ``provenance.json``.
        The hash is computed next. ``dataset.json`` is last.
    """
    n, _channels, height, width = _require_array_shapes(
        images,
        masks,
        labels,
        band_count=len(provenance.band_names),
    )
    if group_ids is not None and splits is not None:
        raise ValueError("pass group_ids or splits, not both")
    if splits is not None and recipe is not None:
        raise ValueError("recipe applies only when group_ids is passed")
    if splits is None:
        if group_ids is None:
            raise ValueError("pass group_ids or splits")
        if len(group_ids) != n:
            raise ValueError(f"len(group_ids) must equal N; got {len(group_ids)} and {n}")
        index = assign_group_splits(group_ids, recipe if recipe is not None else SplitRecipe())
    else:
        index = splits
    _require_complete_split(index, n)
    root = Path(dest)
    root.mkdir(parents=True, exist_ok=True)
    # np.ndarray[float32, (N, C, H, W)]
    np.save(root / "images.npy", np.ascontiguousarray(images))
    # np.ndarray[float32, (N, 1, H, W)]
    np.save(root / "masks.npy", np.ascontiguousarray(masks))
    # np.ndarray[float32, (N, 1)]
    np.save(root / "labels.npy", np.ascontiguousarray(labels))
    write_splits(root / "splits.json", index)
    write_provenance(root / "provenance.json", provenance)
    digest = compute_dataset_hash(root)
    meta = dataset_meta_from_provenance(
        provenance,
        dataset_hash=digest,
        source_doi=source_doi,
        n=n,
        height=height,
        width=width,
    )
    write_dataset_meta(root / "dataset.json", meta)
    return meta


def load_processed_pack(dest: str | Path) -> ProcessedPack:
    """Load a processed pack and check the hash and shapes.

    Args:
        dest: Pack directory.

    Returns:
        ProcessedPack: Memmap arrays, splits, and meta.

    Raises:
        ValueError: If the hash does not match, provenance disagrees with the
            meta, or an array shape disagrees with the meta.
        FileNotFoundError: If a pack file is missing.
        OSError / json.JSONDecodeError: On a malformed sidecar.
    """
    root = Path(dest)
    meta = load_dataset_meta(root / "dataset.json", pack_dir=root)
    provenance = load_provenance(root / "provenance.json")
    if provenance != provenance_from_meta(meta):
        raise ValueError("provenance.json does not match dataset.json")
    images = np.load(root / "images.npy", mmap_mode="r", allow_pickle=False)
    masks = np.load(root / "masks.npy", mmap_mode="r", allow_pickle=False)
    labels = np.load(root / "labels.npy", mmap_mode="r", allow_pickle=False)
    splits = load_splits(root / "splits.json")
    pack = ProcessedPack(images=images, masks=masks, labels=labels, splits=splits, meta=meta)
    _require_consistent_pack(pack)
    return pack


def assert_same_ingest(packs: Sequence[ProcessedPack]) -> None:
    """Raise when pack ingest paths differ.

    Args:
        packs: Packs to compare.

    Returns:
        None.

    Raises:
        ValueError: If ``packs`` is empty or ``ingest_path`` is not identical.
    """
    if not packs:
        raise ValueError("need at least one pack")
    ingest = packs[0].meta.ingest_path
    for pack in packs[1:]:
        if pack.meta.ingest_path != ingest:
            raise ValueError(f"ingest_path mismatch: {ingest!r} != {pack.meta.ingest_path!r}")


def concat_packs(packs: Sequence[ProcessedPack]) -> ProcessedPack:
    """Stack packs that share provenance, bands, norm, and spatial size.

    Args:
        packs: Packs in concatenation order. Each pack's arrays match its meta.

    Returns:
        ProcessedPack: Images, masks, and labels stacked on N. Split indices are
        shifted by the preceding sample count. Provenance is the shared
        provenance. ``dataset_hash`` is empty. ``n`` is the sum of the inputs.

    Raises:
        ValueError: If ``ingest_path`` differs, or band names, spatial size,
            norm, or another provenance field differs.

    Notes:
        This function does not write a directory. The empty hash is meaningful
        only after ``write_processed_pack`` rewrites the files.
    """
    assert_same_ingest(packs)
    for pack in packs:
        _require_consistent_pack(pack)
    _require_same_layout(packs)
    # np.ndarray[float32, (N, C, H, W)]
    images = np.concatenate([np.asarray(pack.images) for pack in packs], axis=0)
    # np.ndarray[float32, (N, 1, H, W)]
    masks = np.concatenate([np.asarray(pack.masks) for pack in packs], axis=0)
    # np.ndarray[float32, (N, 1)]
    labels = np.concatenate([np.asarray(pack.labels) for pack in packs], axis=0)
    train: list[int] = []
    val: list[int] = []
    test: list[int] = []
    offset = 0
    for pack in packs:
        train.extend(item + offset for item in pack.splits.train)
        val.extend(item + offset for item in pack.splits.val)
        test.extend(item + offset for item in pack.splits.test)
        offset += pack.meta.n
    first = packs[0].meta
    meta = dataset_meta_from_provenance(
        provenance_from_meta(first),
        dataset_hash="",
        source_doi=first.source_doi,
        n=int(images.shape[0]),
        height=first.height,
        width=first.width,
    )
    return ProcessedPack(
        images=images,
        masks=masks,
        labels=labels,
        splits=SplitIndex(train=tuple(train), val=tuple(val), test=tuple(test)),
        meta=meta,
    )


def _require_same_layout(packs: Sequence[ProcessedPack]) -> None:
    """Raise when packs do not share layout and provenance.

    Args:
        packs: Non-empty sequence with one ingest path.

    Returns:
        None.

    Raises:
        ValueError: If band names, spatial size, norm, source DOI, or another
            provenance field differs.
    """
    first_meta = packs[0].meta
    first = provenance_from_meta(first_meta)
    for pack in packs[1:]:
        other = provenance_from_meta(pack.meta)
        if other.band_names != first.band_names:
            raise ValueError(f"band_names mismatch: {first.band_names!r} != {other.band_names!r}")
        if pack.meta.height != first_meta.height or pack.meta.width != first_meta.width:
            raise ValueError(
                "spatial size mismatch: "
                f"{first_meta.height}x{first_meta.width} != {pack.meta.height}x{pack.meta.width}"
            )
        if other.norm != first.norm:
            raise ValueError(f"norm mismatch: {first.norm!r} != {other.norm!r}")
        if other != first:
            raise ValueError("provenance mismatch")
        if pack.meta.source_doi != first_meta.source_doi:
            raise ValueError(
                f"source_doi mismatch: {first_meta.source_doi!r} != {pack.meta.source_doi!r}"
            )


def _require_consistent_pack(pack: ProcessedPack) -> None:
    """Raise when arrays or splits disagree with the meta.

    Args:
        pack: In-memory pack.

    Returns:
        None.

    Raises:
        ValueError: If dtype, shape, channel count, or split coverage is wrong.
    """
    n, channels, height, width = _require_array_shapes(
        pack.images,
        pack.masks,
        pack.labels,
        band_count=len(pack.meta.band_names),
    )
    if (
        n != pack.meta.n
        or channels != pack.meta.in_channels
        or height != pack.meta.height
        or width != pack.meta.width
    ):
        raise ValueError(
            f"array shape {(n, channels, height, width)} does not match meta "
            f"{(pack.meta.n, pack.meta.in_channels, pack.meta.height, pack.meta.width)}"
        )
    if channels != len(pack.meta.band_names) or channels != pack.meta.in_channels:
        raise ValueError(
            f"images channels {channels} must equal in_channels {pack.meta.in_channels} "
            f"and len(band_names) {len(pack.meta.band_names)}"
        )
    _require_complete_split(pack.splits, n)


def _require_array_shapes(
    images: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    *,
    band_count: int,
) -> tuple[int, int, int, int]:
    """Return ``(N, C, H, W)`` after checking pack array contracts.

    Args:
        images: Candidate image tensor.
        masks: Candidate mask tensor.
        labels: Candidate label tensor.
        band_count: Expected channel count.

    Returns:
        tuple[int, int, int, int]: ``N``, ``C``, ``H``, and ``W``.

    Raises:
        ValueError: If a dtype is not float32 or a shape does not match.
    """
    _require_float32("images", images)
    _require_float32("masks", masks)
    _require_float32("labels", labels)
    if images.ndim != 4:
        raise ValueError(f"images must have shape (N, C, H, W); got {images.shape}")
    n = int(images.shape[0])
    channels = int(images.shape[1])
    height = int(images.shape[2])
    width = int(images.shape[3])
    if n < 1 or channels < 1 or height < 1 or width < 1:
        raise ValueError(f"images shape {images.shape} must have positive N, C, H, and W")
    if channels != band_count:
        raise ValueError(f"images channels {channels} must equal len(band_names) {band_count}")
    if masks.shape != (n, 1, height, width):
        raise ValueError(f"masks must have shape {(n, 1, height, width)}; got {masks.shape}")
    if labels.shape != (n, 1):
        raise ValueError(f"labels must have shape {(n, 1)}; got {labels.shape}")
    return n, channels, height, width


def _require_float32(name: str, array: np.ndarray) -> None:
    """Raise when ``array`` is not float32.

    Args:
        name: Array role, used in the error.
        array: Candidate tensor.

    Returns:
        None.

    Raises:
        ValueError: If the dtype is not float32.
    """
    if array.dtype != np.float32:
        raise ValueError(f"{name} dtype must be float32; got {array.dtype}")


def _require_complete_split(index: SplitIndex, n: int) -> None:
    """Raise when indices are out of range, overlap, or skip a row.

    Args:
        index: Train, val, and test indices.
        n: Sample count.

    Returns:
        None.

    Raises:
        ValueError: If an index is out of range, duplicated, or the union is
            not ``0..n-1``.
    """
    seen: set[int] = set()
    for name in ("train", "val", "test"):
        for item in index.for_name(name):
            if item < 0 or item >= n:
                raise ValueError(f"split index {item} out of range for n={n}")
            if item in seen:
                raise ValueError(f"duplicate index {item} in splits")
            seen.add(item)
    if len(seen) != n:
        raise ValueError(f"splits must cover 0..{n - 1}")
