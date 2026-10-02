"""Build a finished dataset from any raw tile source.

Contains:
  - build_dataset: validate, split, normalize, augment, and write shards.
  - build_flight, build_zenodo: source-specific wrappers.

The build writes a sibling temporary directory and renames it onto ``dest``
only after ``dataset.json`` is in place. A failure removes the temporary
directory and leaves ``dest`` absent.
"""

from __future__ import annotations

import math
import os
import shutil
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from tools.ml_models.dataset.augment import apply_dihedral, legal_elements
from tools.ml_models.dataset.geometry import tile_hw
from tools.ml_models.dataset.manifest import (
    SCHEMA_VERSION,
    BinRecord,
    DatasetManifest,
    ShardCount,
    compute_dataset_hash,
    write_manifest,
)
from tools.ml_models.dataset.preprocess import IMAGE_SCALE, quantize_unit, to_unit
from tools.ml_models.dataset.raw import BinSpec, RawSource, RawTile, RawTileRef
from tools.ml_models.dataset.sources.flight import FlightTileDir
from tools.ml_models.dataset.spec import BuildSpec
from tools.ml_models.dataset.split import assign_group_splits
from tools.ml_models.dataset.store import RowRecord, ShardWriter

_POSITIVE = 0.5


@dataclass(frozen=True, slots=True)
class _Planned:
    """One output row before pixels are read.

    Attributes:
        index: Position in the source index.
        task: ``classifier`` or ``segmentor``.
        split: ``train``, ``val``, or ``test``.
        element: Dihedral name.
        height: Output H.
        width: Output W.
        lateral_m: Stored lateral GSD.
        along_m: Stored along-track GSD.
        label: Classification target copied from the source row.
    """

    index: int
    task: str
    split: str
    element: str
    height: int
    width: int
    lateral_m: float
    along_m: float
    label: float


def build_dataset(source: RawSource, dest: str | Path, spec: BuildSpec) -> DatasetManifest:
    """Turn one raw source into a finished dataset.

    Args:
        source: Forward-only tile stream.
        dest: Destination directory. Must not already exist.
        spec: Split, augmentation, tasks, and band contract.

    Returns:
        DatasetManifest: Identity written to ``dataset.json``.

    Raises:
        FileExistsError: If ``dest`` already exists.
        ValueError: If bands, GSD, shape, or the tile stream break the contract.
        OSError: If the destination cannot be created.
    """
    root = Path(dest)
    if root.exists():
        raise FileExistsError(f"dataset destination exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.partial-", dir=root.parent))
    try:
        manifest = _build_into(source, temporary, spec)
        if root.exists():
            raise FileExistsError(f"dataset destination exists: {root}")
        os.rename(temporary, root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def build_flight(
    source_dir: str | Path,
    dest: str | Path,
    spec: BuildSpec | None = None,
) -> DatasetManifest:
    """Build a dataset from a labeled flight tile directory.

    Args:
        source_dir: Directory with ``source.json``, ``index.jsonl``, and ``tiles/``.
        dest: Finished dataset directory.
        spec: Build specification. The default spec is used when None.

    Returns:
        DatasetManifest: Written identity.

    Raises:
        FileExistsError: If ``dest`` already exists.
        ValueError: If the flight ``gsd_reference_m`` disagrees with ``spec``,
            or the raw contract fails.
    """
    source = FlightTileDir(source_dir)
    if any(ref.gsd_nominal for ref in source.index()):
        raise ValueError("nominal GSD rows cannot enter the standard flight dataset")
    resolved = BuildSpec() if spec is None else spec
    if source.gsd_reference_m != resolved.gsd_reference_m:
        raise ValueError(
            f"flight gsd_reference_m {source.gsd_reference_m} != spec {resolved.gsd_reference_m}"
        )
    return build_dataset(source, dest, resolved)


def build_zenodo(
    images_tar: str | Path,
    labels_tar: str | Path,
    weights_path: str | Path,
    dest: str | Path,
    spec: BuildSpec | None = None,
    bins: tuple[BinSpec, ...] | None = None,
) -> DatasetManifest:
    """Build a dataset from the Zenodo 4250706 archives.

    Args:
        images_tar: Image archive of class-directory GeoTIFFs.
        labels_tar: Label Studio annotation archive.
        weights_path: Prism weight table TOML.
        dest: Finished dataset directory.
        spec: Build specification. The default spec is used when None.
        bins: GSD bins to emit. ``DEFAULT_BINS`` when None.

    Returns:
        DatasetManifest: Written identity.

    Raises:
        FileExistsError: If ``dest`` already exists.
        FileNotFoundError: If an archive or the weight table is missing.
        ValueError: If ``spec.weight_table_id`` names a different table, or
            the raw contract fails.
    """
    from tools.ml_models.dataset.sources.zenodo.adapt import ZenodoSource
    from tools.ml_models.dataset.sources.zenodo.bins import DEFAULT_BINS
    from tools.ml_models.dataset.sources.zenodo.prism import load_weight_table

    source = ZenodoSource(
        images_tar,
        labels_tar,
        load_weight_table(weights_path),
        DEFAULT_BINS if bins is None else bins,
    )
    resolved = BuildSpec() if spec is None else spec
    if resolved.weight_table_id and resolved.weight_table_id != source.weight_table_id:
        raise ValueError(
            f"spec weight_table_id {resolved.weight_table_id!r} "
            f"!= weight table {source.weight_table_id!r}"
        )
    resolved = replace(resolved, weight_table_id=source.weight_table_id)
    return build_dataset(source, dest, resolved)


def _build_into(source: RawSource, dest: Path, spec: BuildSpec) -> DatasetManifest:
    """Write shards and ``dataset.json`` under an existing empty directory.

    Args:
        source: Raw tile stream.
        dest: Empty directory that becomes the dataset root.
        spec: Build specification.

    Returns:
        DatasetManifest: Identity whose hash matches the files just written.

    Raises:
        ValueError: If the contract fails or no rows are selected.
    """
    _require_bands(source, spec)
    refs = source.index()
    if len(refs) < 1:
        raise ValueError("source index is empty")
    _require_unique_ids(refs)
    heights, widths, laterals, alongs = _geometry_table(source, refs)
    split_names = _split_names(refs, spec)
    planned = _plan(refs, spec, split_names, heights, widths, laterals, alongs)
    if not planned:
        raise ValueError("no rows selected for the requested tasks")
    writers = _open_writers(dest, planned)
    try:
        _fill(source, refs, planned, heights, widths, writers)
        for writer in writers.values():
            writer.close()
    except Exception:
        for writer in writers.values():
            writer.abort()
        raise
    manifest = _manifest(source, spec, planned, compute_dataset_hash(dest))
    write_manifest(dest / "dataset.json", manifest)
    return manifest


def _require_bands(source: RawSource, spec: BuildSpec) -> None:
    """Raise ValueError when the source bands differ from the spec.

    Args:
        source: Raw source.
        spec: Build specification.

    Returns:
        None.

    Raises:
        ValueError: On a band mismatch or an unknown domain.
    """
    if tuple(source.band_names) != tuple(spec.input_bands):
        raise ValueError(
            f"band_names {tuple(source.band_names)} != input_bands {tuple(spec.input_bands)}"
        )
    if source.domain not in ("dn", "unit"):
        raise ValueError(f"unknown domain {source.domain!r}")


def _require_unique_ids(refs: tuple[RawTileRef, ...]) -> None:
    """Raise ValueError when a tile id repeats or a group id is empty.

    Args:
        refs: Source index.

    Returns:
        None.

    Raises:
        ValueError: On a duplicate tile id or an empty group id.
    """
    seen: set[str] = set()
    for ref in refs:
        if not ref.tile_id or ref.tile_id in seen:
            raise ValueError(f"tile_id must be unique and non-empty; got {ref.tile_id!r}")
        seen.add(ref.tile_id)
        if not ref.group_id:
            raise ValueError(f"group_id must be non-empty for {ref.tile_id}")


def _geometry_table(
    source: RawSource,
    refs: tuple[RawTileRef, ...],
) -> tuple[list[int], list[int], list[float], list[float]]:
    """Return expected H, W, and stored GSD for each index row.

    Args:
        source: Raw source. ``extent_m`` selects the size rule.
        refs: Source index.

    Returns:
        tuple[list[int], list[int], list[float], list[float]]: Heights, widths,
        lateral metres, and along-track metres.

    Raises:
        ValueError: If a GSD component is not finite and positive, or the
            rounded size is below 1.
    """
    heights: list[int] = []
    widths: list[int] = []
    laterals: list[float] = []
    alongs: list[float] = []
    for ref in refs:
        _require_gsd(ref)
        if not math.isfinite(ref.label) or ref.label not in (0.0, 1.0):
            raise ValueError(f"label must be binary for {ref.tile_id}")
        height, width = _expected_hw(source, ref)
        lateral, along = _stored_gsd(source, ref, height, width)
        heights.append(height)
        widths.append(width)
        laterals.append(lateral)
        alongs.append(along)
    return heights, widths, laterals, alongs


def _require_gsd(ref: RawTileRef) -> None:
    """Raise ValueError when a tile GSD is not finite and positive.

    Args:
        ref: Source row.

    Returns:
        None.

    Raises:
        ValueError: If either component fails the check.
    """
    for name, value in (("lateral", ref.gsd.lateral_m), ("along", ref.gsd.along_m)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} gsd must be finite and > 0 for {ref.tile_id}; got {value}")


def _expected_hw(source: RawSource, ref: RawTileRef) -> tuple[int, int]:
    """Return the pixel size required for one tile.

    Args:
        source: Raw source.
        ref: Source row.

    Returns:
        tuple[int, int]: ``(height, width)``.

    Raises:
        ValueError: If ``extent_m`` rounds to an empty tile.

    Notes:
        A missing extent means the flight tile size. A present extent is
        ``(lateral_m, along_m)`` and the size is ``round(extent / gsd)``.
    """
    if source.extent_m is None:
        return tile_hw()
    lateral_extent, along_extent = source.extent_m
    if any(not math.isfinite(value) or value <= 0 for value in source.extent_m):
        raise ValueError(f"extent_m must be positive; got {source.extent_m}")
    width = int(round(lateral_extent / ref.gsd.lateral_m))
    height = int(round(along_extent / ref.gsd.along_m))
    if height < 1 or width < 1:
        raise ValueError(f"rounded size for {ref.tile_id} is {height}x{width}")
    return (height, width)


def _stored_gsd(source: RawSource, ref: RawTileRef, height: int, width: int) -> tuple[float, float]:
    """Return the GSD written to ``gsd.npy``.

    Args:
        source: Raw source.
        ref: Source row.
        height: Expected H.
        width: Expected W.

    Returns:
        tuple[float, float]: Lateral and along-track metres. When the source
        has an extent, the values are ``extent / pixels`` after rounding.
    """
    if source.extent_m is None:
        return (ref.gsd.lateral_m, ref.gsd.along_m)
    lateral_extent, along_extent = source.extent_m
    return (lateral_extent / width, along_extent / height)


def _split_names(refs: tuple[RawTileRef, ...], spec: BuildSpec) -> list[str]:
    """Assign one split name per index row.

    Args:
        refs: Source index.
        spec: Build specification.

    Returns:
        list[str]: ``train``, ``val``, or ``test`` for each row. Rows that
        share a group share a name.
    """
    index = assign_group_splits([ref.group_id for ref in refs], spec.split)
    names = [""] * len(refs)
    for split_name in ("train", "val", "test"):
        for row in index.for_name(split_name):
            names[row] = split_name
    return names


def _plan(
    refs: tuple[RawTileRef, ...],
    spec: BuildSpec,
    split_names: list[str],
    heights: list[int],
    widths: list[int],
    laterals: list[float],
    alongs: list[float],
) -> list[_Planned]:
    """Expand index rows into output rows, including train elements.

    Args:
        refs: Source index.
        spec: Build specification.
        split_names: Split name per index row.
        heights: Expected H per index row.
        widths: Expected W per index row.
        laterals: Stored lateral GSD per index row.
        alongs: Stored along-track GSD per index row.

    Returns:
        list[_Planned]: Output rows in stream order, tasks then elements.
    """
    planned: list[_Planned] = []
    for index, ref in enumerate(refs):
        height = heights[index]
        width = widths[index]
        elements = _elements_for(split_names[index], height, width, spec)
        if not math.isclose(laterals[index], alongs[index], rel_tol=1e-6):
            elements = tuple(
                element for element in elements if element in ("id", "rot180", "flip_h", "flip_v")
            )
            if not elements:
                raise ValueError(f"no axis-preserving augmentation for {ref.tile_id}")
        for task in spec.tasks:
            if task == "segmentor" and not ref.has_mask:
                continue
            for element in elements:
                planned.append(
                    _Planned(
                        index=index,
                        task=task,
                        split=split_names[index],
                        element=element,
                        height=height,
                        width=width,
                        lateral_m=laterals[index],
                        along_m=alongs[index],
                        label=ref.label,
                    )
                )
    return planned


def _elements_for(split_name: str, height: int, width: int, spec: BuildSpec) -> tuple[str, ...]:
    """Return the dihedral elements written for one tile.

    Args:
        split_name: ``train``, ``val``, or ``test``.
        height: Tile H.
        width: Tile W.
        spec: Build specification.

    Returns:
        tuple[str, ...]: ``("id",)`` outside train. On train, the recipe
        intersected with ``legal_elements``.

    Raises:
        ValueError: If the train intersection is empty.
    """
    if split_name != "train":
        return ("id",)
    legal = set(legal_elements(height, width))
    chosen = tuple(name for name in spec.augment.elements if name in legal)
    if not chosen:
        raise ValueError(f"no legal augment elements for {height}x{width}")
    return chosen


def _open_writers(
    dest: Path, planned: list[_Planned]
) -> dict[tuple[str, str, int, int], ShardWriter]:
    """Allocate one writer per shard.

    Args:
        dest: Dataset root.
        planned: Output rows.

    Returns:
        dict[tuple[str, str, int, int], ShardWriter]: Keyed by task, split, H, W.
    """
    counts: dict[tuple[str, str, int, int], int] = {}
    for item in planned:
        key = (item.task, item.split, item.height, item.width)
        counts[key] = counts.get(key, 0) + 1
    writers: dict[tuple[str, str, int, int], ShardWriter] = {}
    for key, count in counts.items():
        task, split_name, height, width = key
        directory = dest / task / split_name / f"{height}x{width}"
        writers[key] = ShardWriter(
            directory,
            count,
            height,
            width,
            with_masks=task == "segmentor",
        )
    return writers


def _fill(
    source: RawSource,
    refs: tuple[RawTileRef, ...],
    planned: list[_Planned],
    heights: list[int],
    widths: list[int],
    writers: dict[tuple[str, str, int, int], ShardWriter],
) -> None:
    """Stream tiles once and append every planned row.

    Args:
        source: Raw source.
        refs: Source index.
        planned: Output rows in stream order.
        heights: Expected H per index row.
        widths: Expected W per index row.
        writers: Open shard writers.

    Returns:
        None.

    Raises:
        ValueError: If the stream order, shape, or mask flag disagrees with
            the index.
    """
    by_index: dict[int, list[_Planned]] = {}
    for item in planned:
        by_index.setdefault(item.index, []).append(item)
    seen = 0
    for index, tile in enumerate(source.iter_tiles()):
        if index >= len(refs):
            raise ValueError("iter_tiles yielded more rows than index")
        _require_tile(tile, refs[index], heights[index], widths[index], source)
        unit = to_unit(tile.image, source.domain, source.bit_depth)
        for item in by_index.get(index, []):
            _append_planned(writers, item, tile, unit)
        seen += 1
    if seen != len(refs):
        raise ValueError("iter_tiles ended before the index")


def _require_tile(
    tile: RawTile,
    ref: RawTileRef,
    height: int,
    width: int,
    source: RawSource,
) -> None:
    """Raise ValueError when a streamed tile disagrees with its index row.

    Args:
        tile: Streamed tile.
        ref: Index row at the same position.
        height: Expected H.
        width: Expected W.
        source: Raw source, used for the channel count.

    Returns:
        None.

    Raises:
        ValueError: On an id, shape, or mask mismatch.
    """
    if tile.ref != ref:
        raise ValueError(f"iter_tiles order mismatch at {ref.tile_id}: got {tile.ref.tile_id}")
    expected = (len(source.band_names), height, width)
    if tile.image.shape != expected:
        raise ValueError(
            f"{ref.tile_id} shape {tile.image.shape} is inconsistent with gsd size {expected}"
        )
    if ref.has_mask:
        if tile.mask is None or tile.mask.shape != (1, height, width):
            raise ValueError(f"{ref.tile_id} mask must be (1, {height}, {width})")
        if not np.all(np.isfinite(tile.mask)) or not np.all((tile.mask == 0) | (tile.mask == 1)):
            raise ValueError(f"{ref.tile_id} mask must contain binary pixels")
    elif tile.mask is not None:
        raise ValueError(f"{ref.tile_id} has a mask but has_mask is false")
    if not np.all(np.isfinite(tile.image)):
        raise ValueError(f"{ref.tile_id} image contains non-finite pixels")


def _append_planned(
    writers: dict[tuple[str, str, int, int], ShardWriter],
    item: _Planned,
    tile: RawTile,
    unit: np.ndarray,
) -> None:
    """Quantize one augmented row into its shard.

    Args:
        writers: Open shard writers.
        item: Planned output row.
        tile: Source tile.
        unit: np.ndarray[float32, (3, H, W)] in ``[0, 1]``.

    Returns:
        None.
    """
    image = quantize_unit(apply_dihedral(unit, item.element))
    mask: np.ndarray | None = None
    if item.task == "segmentor":
        if tile.mask is None:
            raise ValueError(f"{tile.ref.tile_id} segmentor row is missing a mask")
        transformed = apply_dihedral(tile.mask, item.element)
        mask = transformed.astype(np.uint8)
    if tile.ref.theta_g_deg is not None and not math.isfinite(tile.ref.theta_g_deg):
        raise ValueError(f"{tile.ref.tile_id} theta_g_deg must be finite")
    row = RowRecord(
        tile_id=tile.ref.tile_id,
        group_id=tile.ref.group_id,
        frame_id=tile.ref.frame_id,
        grid_rc=tile.ref.grid_rc,
        bin_id=tile.ref.bin_id,
        element=item.element,
        theta_g_deg=tile.ref.theta_g_deg,
        gsd_nominal=tile.ref.gsd_nominal,
    )
    key = (item.task, item.split, item.height, item.width)
    writers[key].append(
        image,
        np.asarray([item.lateral_m, item.along_m], dtype=np.float32),
        tile.ref.label,
        mask,
        row,
    )


def _manifest(
    source: RawSource,
    spec: BuildSpec,
    planned: list[_Planned],
    dataset_hash: str,
) -> DatasetManifest:
    """Build the in-memory manifest for a completed directory.

    Args:
        source: Raw source.
        spec: Build specification.
        planned: Output rows that were written.
        dataset_hash: Content hash of the shard files.

    Returns:
        DatasetManifest: Identity with ``norm`` ``unit`` and ``image_scale`` 65535.
    """
    counts: dict[tuple[str, str, int, int], list[int]] = {}
    laterals = [item.lateral_m for item in planned]
    alongs = [item.along_m for item in planned]
    for item in planned:
        key = (item.task, item.split, item.height, item.width)
        bucket = counts.setdefault(key, [0, 0])
        bucket[0] += 1
        if item.label >= _POSITIVE:
            bucket[1] += 1
    shards = tuple(
        ShardCount(
            task=task,
            split=split_name,
            height=height,
            width=width,
            n=bucket[0],
            n_positive=bucket[1],
        )
        for (task, split_name, height, width), bucket in sorted(counts.items())
    )
    bins = tuple(
        BinRecord(
            bin_id=item.bin_id,
            lateral_m=item.lateral_m,
            along_m=item.along_m,
            elevation_deg=item.elevation_deg,
        )
        for item in source.bins
    )
    return DatasetManifest(
        schema_version=SCHEMA_VERSION,
        source=source.name,
        source_ref=source.source_ref,
        weight_table_id=spec.weight_table_id,
        band_names=tuple(spec.input_bands),
        norm="unit",
        image_scale=IMAGE_SCALE,
        gsd_reference_m=spec.gsd_reference_m,
        split=spec.split,
        augment=spec.augment,
        bins=bins,
        shards=shards,
        gsd_lateral_min_m=min(laterals),
        gsd_lateral_max_m=max(laterals),
        gsd_along_min_m=min(alongs),
        gsd_along_max_m=max(alongs),
        dataset_hash=dataset_hash,
    )
