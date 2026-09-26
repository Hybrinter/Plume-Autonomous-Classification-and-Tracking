"""Zenodo 4250706 archive index, tile cache, and location splits.

Contains:
  - TileRef, TileIndex, location_id_of, build_index, iter_stacks.
  - TileCache, to_native_stack, build_cache, build_mask_cache, open_cache.
  - SidePack, prepare_side, open_side_pack.
  - SplitRecipe, LocationSplit, assign_location_splits.

``assign_location_splits`` calls ``assign_group_splits``. Rasterio is imported
inside the GeoTIFF reader.
"""

from __future__ import annotations

import io
import json
import tarfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tools.ml_models.data.grid import LEGAL_SIDES, NATIVE_SIDE, coarsen, rasterize_mask
from tools.ml_models.data.split import SplitRecipe as GroupSplitRecipe
from tools.ml_models.data.split import assign_group_splits

_GEOTIFF_SUFFIXES = (".tif", ".tiff")
_LABEL_SUFFIX = "_features.json"
_META = "meta.json"
_STACKS = "stacks.dat"
_SLACK = 2


@dataclass(frozen=True, slots=True)
class TileRef:
    """One corpus image.

    Attributes:
        stem: Filename stem shared by the image and its annotation.
        location_id: Leading underscore token of the stem.
        positive: True when the image lives in a ``positive`` directory.
        member_name: Path of the image inside the image archive.
        polygons: Percentage-space ``(V, 2)`` vertices. Empty when the
            annotation file exists and has no smoke polygon. ``None`` when the
            stem has no annotation file.
    """

    stem: str
    location_id: str
    positive: bool
    member_name: str
    polygons: tuple[np.ndarray, ...] | None


@dataclass(frozen=True, slots=True)
class TileIndex:
    """Stems in archive order, one row per stem."""

    tiles: tuple[TileRef, ...]

    def by_stem(self) -> dict[str, TileRef]:
        """Return tiles keyed by stem."""
        return {tile.stem: tile for tile in self.tiles}


def location_id_of(stem: str) -> str:
    """Return the leading underscore token of a stem.

    Args:
        stem: Image stem such as ``10003_2019-01-21T10:56:41.330Z_0``.

    Returns:
        str: The token before the first underscore, or the whole stem when
        there is no underscore.
    """
    return stem.split("_", 1)[0]


def _parse_polygons(payload: object) -> tuple[np.ndarray, ...]:
    """Return smoke polygons from one Label Studio export."""
    if not isinstance(payload, dict):
        return ()
    polygons: list[np.ndarray] = []
    for completion in payload.get("completions", []):
        if not isinstance(completion, dict):
            continue
        for result in completion.get("result", []):
            if not isinstance(result, dict) or result.get("type") != "polygonlabels":
                continue
            value = result.get("value", {})
            if not isinstance(value, dict):
                continue
            labels = value.get("polygonlabels", [])
            if "smoke" not in labels:
                continue
            points = np.asarray(value.get("points", []), dtype=np.float32)
            if points.ndim == 2 and points.shape[0] >= 3:
                polygons.append(points)
    return tuple(polygons)


def _annotation_stem(name: str) -> str:
    """Return the image stem for an annotation member name."""
    filename = Path(name).name
    if filename.endswith(_LABEL_SUFFIX):
        return filename[: -len(_LABEL_SUFFIX)]
    return Path(filename).stem


def build_index(images_tar: Path, labels_tar: Path) -> TileIndex:
    """Index image stems, presence labels, and polygons.

    Args:
        images_tar: Archive of class-directory GeoTIFFs.
        labels_tar: Archive of Label Studio JSON files.

    Returns:
        TileIndex: One tile per image stem. Duplicate stems keep the first image.

    Raises:
        FileNotFoundError: If either archive is missing.
        ValueError: If an image stem has no location token.
    """
    if not images_tar.is_file():
        raise FileNotFoundError(images_tar)
    if not labels_tar.is_file():
        raise FileNotFoundError(labels_tar)
    polygons_by_stem: dict[str, tuple[np.ndarray, ...]] = {}
    with tarfile.open(labels_tar, "r:*") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".json"):
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            payload = json.loads(extracted.read().decode("utf-8"))
            polygons_by_stem[_annotation_stem(member.name)] = _parse_polygons(payload)
    seen: set[str] = set()
    tiles: list[TileRef] = []
    with tarfile.open(images_tar, "r:*") as archive:
        for member in archive:
            if not member.isfile():
                continue
            filename = Path(member.name).name
            if not filename.lower().endswith(_GEOTIFF_SUFFIXES):
                continue
            stem = Path(filename).stem
            if stem in seen:
                continue
            seen.add(stem)
            location = location_id_of(stem)
            if not location:
                raise ValueError(f"stem {stem!r} has an empty location id")
            positive = "positive" in Path(member.name).parts
            polygons = polygons_by_stem.get(stem)
            tiles.append(
                TileRef(
                    stem=stem,
                    location_id=location,
                    positive=positive,
                    member_name=member.name,
                    polygons=polygons,
                )
            )
    return TileIndex(tiles=tuple(tiles))


def _stack_from_geotiff(payload: bytes) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return float32 bands and descriptions from one GeoTIFF payload.

    Args:
        payload: GeoTIFF bytes from one archive member.

    Returns:
        tuple: Float32 array ``(C, H, W)`` and one description per band.
        A missing description is an empty string.

    Raises:
        ImportError: If rasterio is not installed.
    """
    import rasterio

    with rasterio.open(io.BytesIO(payload)) as dataset:
        stack = dataset.read().astype(np.float32)
        descriptions = tuple("" if item is None else str(item) for item in dataset.descriptions)
    return stack, descriptions


def iter_stacks(
    images_tar: Path,
    tiles: Sequence[TileRef],
) -> Iterator[tuple[TileRef, np.ndarray, tuple[str, ...]]]:
    """Yield each requested GeoTIFF from one forward pass of the image archive.

    Args:
        images_tar: Image archive. Gzip and uncompressed tar are both accepted.
        tiles: Tiles to read. Member names that are not in this sequence are
            skipped. An empty sequence yields nothing and does not open the
            archive.

    Returns:
        Iterator: ``(tile, stack, descriptions)`` in archive order. ``stack``
        is float32 ``(C, H, W)``. A missing description is an empty string.

    Raises:
        FileNotFoundError: If the archive is missing, or a requested member is
            absent.
        ImportError: If rasterio is not installed.
    """
    if tiles and not images_tar.is_file():
        raise FileNotFoundError(images_tar)
    return _iter_stacks(images_tar, tiles)


def _iter_stacks(
    images_tar: Path,
    tiles: Sequence[TileRef],
) -> Iterator[tuple[TileRef, np.ndarray, tuple[str, ...]]]:
    """Stream ``tiles`` from ``images_tar`` in archive order.

    Args:
        images_tar: Image archive that ``iter_stacks`` has already checked.
        tiles: Tiles to read. Empty input yields nothing.

    Returns:
        Iterator: ``(tile, stack, descriptions)`` in archive order.

    Raises:
        FileNotFoundError: If a requested member is absent.
        ImportError: If rasterio is not installed.
    """
    if not tiles:
        return
    wanted: dict[str, list[TileRef]] = {}
    for tile in tiles:
        wanted.setdefault(tile.member_name, []).append(tile)
    remaining = len(tiles)
    with tarfile.open(images_tar, "r|*") as archive:
        for member in archive:
            refs = wanted.get(member.name)
            if refs is None or not member.isfile():
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                raise FileNotFoundError(member.name)
            stack, descriptions = _stack_from_geotiff(extracted.read())
            for ref in refs:
                yield ref, stack, descriptions
            del wanted[member.name]
            remaining -= len(refs)
            if remaining == 0:
                break
    if remaining:
        missing = sorted(wanted)[0]
        raise FileNotFoundError(missing)


def to_native_stack(stack: np.ndarray) -> np.ndarray:
    """Return a float32 ``(C, 120, 120)`` stack.

    Args:
        stack: Array ``(C, H, W)``. Each spatial side must be within 2 pixels
            of 120.

    Returns:
        np.ndarray: Float32 stack. A short side is edge-padded. A long side is
        cropped from the origin.

    Raises:
        ValueError: If the array is not three-dimensional or a side is too far
            from 120.
    """
    array = np.asarray(stack, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"expected (C, H, W); got {array.shape}")
    _channels, height, width = array.shape
    if abs(height - NATIVE_SIDE) > _SLACK or abs(width - NATIVE_SIDE) > _SLACK:
        raise ValueError(f"expected (C, 120, 120); got {array.shape}")
    if height == NATIVE_SIDE and width == NATIVE_SIDE:
        return array
    fitted = np.empty((_channels, NATIVE_SIDE, NATIVE_SIDE), dtype=np.float32)
    copy_h = min(height, NATIVE_SIDE)
    copy_w = min(width, NATIVE_SIDE)
    fitted[:, :copy_h, :copy_w] = array[:, :copy_h, :copy_w]
    if height < NATIVE_SIDE:
        fitted[:, height:, :copy_w] = fitted[:, height - 1 : height, :copy_w]
    if width < NATIVE_SIDE:
        fitted[:, :, width:] = fitted[:, :, width - 1 : width]
    return fitted


class TileCache:
    """Native ``(C, 120, 120)`` stacks addressed by stem.

    ``reader`` returns a copy. The memmap stays on disk.
    """

    def __init__(self, path: Path, stems: Sequence[str], descriptions: Sequence[str]) -> None:
        """Open an existing memmap.

        Args:
            path: Directory that holds ``meta.json`` and ``stacks.dat``.
            stems: Row order of the memmap.
            descriptions: GeoTIFF band descriptions shared by every stack.

        Raises:
            FileNotFoundError: If the memmap is missing.
            ValueError: If the memmap shape does not match the sidecar.
        """
        data_path = path / _STACKS
        if not data_path.is_file():
            raise FileNotFoundError(data_path)
        self.path = path
        self.stems = tuple(stems)
        self.descriptions = tuple(descriptions)
        self._index = {stem: row for row, stem in enumerate(self.stems)}
        self._stacks = np.memmap(data_path, dtype=np.float32, mode="r")
        channels = 0
        if self.descriptions:
            expected = len(self.stems) * len(self.descriptions) * NATIVE_SIDE * NATIVE_SIDE
            if self._stacks.size != expected:
                raise ValueError(f"cache has {self._stacks.size} values; expected {expected}")
            channels = len(self.descriptions)
        self._shape = (len(self.stems), channels, NATIVE_SIDE, NATIVE_SIDE)

    def reader(self, tile: TileRef) -> np.ndarray:
        """Return one native stack.

        Args:
            tile: Stem that was stored in this cache.

        Returns:
            np.ndarray: Float32 copy ``(C, 120, 120)``.

        Raises:
            KeyError: If ``tile.stem`` is absent.
        """
        row = self._index[tile.stem]
        view = self._stacks.reshape(self._shape)[row]
        return np.array(view, dtype=np.float32, copy=True)

    def close(self) -> None:
        """Release the native memmap."""
        mapped = getattr(self._stacks, "_mmap", None)
        if mapped is not None:
            mapped.close()


def build_mask_cache(
    tiles: Sequence[TileRef], path: Path, side_px: int = NATIVE_SIDE
) -> np.ndarray:
    """Rasterize each annotated tile once and store a uint8 mask plane.

    Args:
        tiles: Corpus tiles in cache row order.
        path: Destination ``.dat`` file. Parent directories are created.
        side_px: Legal output side. The default is 120.

    Returns:
        np.ndarray: Memmap ``(N, side_px, side_px)`` with 1 on smoke and 0 elsewhere.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    masks = np.memmap(path, dtype=np.uint8, mode="w+", shape=(len(tiles), side_px, side_px))
    masks[:] = 0
    for row, tile in enumerate(tiles):
        if tile.polygons:
            masks[row] = rasterize_mask(tile.polygons, side_px, rule="half")[0]
    masks.flush()
    return masks


def open_cache(path: Path) -> TileCache:
    """Reopen a cache directory.

    Args:
        path: Directory written by :func:`build_cache`.

    Returns:
        TileCache: The stored stems and descriptions.

    Raises:
        FileNotFoundError: If the sidecar is missing.
    """
    meta_path = path / _META
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    return TileCache(path, payload["stems"], payload["descriptions"])


def build_cache(images_tar: Path, tiles: Sequence[TileRef], path: Path) -> TileCache:
    """Write one memmap from a single forward pass of the image archive.

    Args:
        images_tar: GeoTIFF archive.
        tiles: Tiles to store. Row order follows this sequence.
        path: Destination directory. Parent directories are created.

    Returns:
        TileCache: The written cache.

    Raises:
        ValueError: If ``tiles`` is empty or a stack is not near ``(C, 120, 120)``.
        FileNotFoundError: If the archive or a member is missing.
    """
    if not tiles:
        raise ValueError("cache needs at least one tile")
    path.mkdir(parents=True, exist_ok=True)
    stems = [tile.stem for tile in tiles]
    index = {stem: row for row, stem in enumerate(stems)}
    descriptions: tuple[str, ...] = ()
    memmap: np.memmap | None = None
    written = 0
    for tile, stack, band_descriptions in iter_stacks(images_tar, tiles):
        array = to_native_stack(stack)
        if memmap is None:
            descriptions = tuple(band_descriptions)
            memmap = np.memmap(
                path / _STACKS,
                dtype=np.float32,
                mode="w+",
                shape=(len(tiles), array.shape[0], NATIVE_SIDE, NATIVE_SIDE),
            )
        elif array.shape[0] != len(descriptions):
            raise ValueError(f"stack channels {array.shape[0]} != {len(descriptions)}")
        memmap[index[tile.stem]] = array
        written += 1
    if memmap is None or written != len(tiles):
        raise ValueError(f"wrote {written} stacks; expected {len(tiles)}")
    memmap.flush()
    del memmap
    payload = {"stems": stems, "descriptions": list(descriptions)}
    (path / _META).write_text(json.dumps(payload), encoding="utf-8")
    return open_cache(path)


@dataclass(frozen=True, slots=True)
class SidePack:
    """One ground-sample size, already resampled.

    Attributes:
        path: Pack directory.
        side_px: Output side.
        stems: Row order shared by the image and mask arrays.
        images: Float32 memmap ``(N, C, side, side)``.
        masks: Uint8 memmap ``(N, side, side)``.
        positive: Uint8 presence label per row.
        annotated: Uint8 annotation flag per row.
    """

    path: Path
    side_px: int
    stems: tuple[str, ...]
    images: np.ndarray
    masks: np.ndarray
    positive: np.ndarray
    annotated: np.ndarray

    def close(self) -> None:
        """Release the image and mask memmaps."""
        for array in (self.images, self.masks, self.positive, self.annotated):
            mapped = getattr(array, "_mmap", None)
            if mapped is not None:
                mapped.close()


def open_side_pack(path: Path) -> SidePack:
    """Open a prepared side pack.

    Args:
        path: Directory written by :func:`prepare_side`.

    Returns:
        SidePack: The stored arrays.

    Raises:
        FileNotFoundError: If the sidecar is missing.
        ValueError: If a memmap does not match the sidecar.
    """
    meta_path = path / _META
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    stems = tuple(str(item) for item in payload["stems"])
    side = int(payload["side_px"])
    channels = int(payload["channels"])
    count = len(stems)
    images = np.memmap(path / "images.dat", dtype=np.float32, mode="r")
    masks = np.memmap(path / "masks.dat", dtype=np.uint8, mode="r")
    positive = np.memmap(path / "positive.dat", dtype=np.uint8, mode="r")
    annotated = np.memmap(path / "annotated.dat", dtype=np.uint8, mode="r")
    expected_images = count * channels * side * side
    if images.size != expected_images or masks.size != count * side * side:
        raise ValueError(f"pack at {path} does not match its sidecar")
    if positive.size != count or annotated.size != count:
        raise ValueError(f"pack flags at {path} do not match its sidecar")
    return SidePack(
        path=path,
        side_px=side,
        stems=stems,
        images=images.reshape(count, channels, side, side),
        masks=masks.reshape(count, side, side),
        positive=positive,
        annotated=annotated,
    )


def prepare_side(
    cache: TileCache,
    tiles: Sequence[TileRef],
    side_px: int,
    path: Path,
    *,
    mask_source: Path | None = None,
) -> SidePack:
    """Resample the native cache once and write a side pack.

    Args:
        cache: Native stacks in the same row order as ``tiles``.
        tiles: Corpus tiles. Presence and annotation flags are copied from here.
        side_px: Legal output side.
        path: Destination directory.
        mask_source: Existing uint8 mask memmap ``(N, side, side)``. ``None``
            rasterizes each annotated tile.

    Returns:
        SidePack: The opened pack. The native cache is left open.

    Raises:
        ValueError: If the side is illegal or the stem order disagrees.
    """
    if side_px not in LEGAL_SIDES:
        raise ValueError(f"side_px must be one of {sorted(LEGAL_SIDES)}; got {side_px}")
    if len(tiles) != len(cache.stems):
        raise ValueError(f"tiles {len(tiles)} != cache rows {len(cache.stems)}")
    for row, tile in enumerate(tiles):
        if tile.stem != cache.stems[row]:
            raise ValueError(f"stem mismatch at row {row}")
    path.mkdir(parents=True, exist_ok=True)
    count = len(tiles)
    probe = coarsen(cache.reader(tiles[0]), side_px)
    channels = int(probe.shape[0])
    images = np.memmap(
        path / "images.dat",
        dtype=np.float32,
        mode="w+",
        shape=(count, channels, side_px, side_px),
    )
    masks = np.memmap(
        path / "masks.dat", dtype=np.uint8, mode="w+", shape=(count, side_px, side_px)
    )
    positive = np.memmap(path / "positive.dat", dtype=np.uint8, mode="w+", shape=(count,))
    annotated = np.memmap(path / "annotated.dat", dtype=np.uint8, mode="w+", shape=(count,))
    copied_masks: np.ndarray | None = None
    if mask_source is not None and mask_source.is_file():
        copied_masks = np.memmap(mask_source, dtype=np.uint8, mode="r")
        if copied_masks.size != count * side_px * side_px:
            copied_masks = None
        else:
            masks[:] = copied_masks.reshape(count, side_px, side_px)
    for row, tile in enumerate(tiles):
        if row == 0:
            images[0] = probe
        else:
            images[row] = coarsen(cache.reader(tile), side_px)
        positive[row] = 1 if tile.positive else 0
        annotated[row] = 0 if tile.polygons is None else 1
        if copied_masks is None and tile.polygons:
            masks[row] = rasterize_mask(tile.polygons, side_px, rule="half")[0]
    images.flush()
    masks.flush()
    positive.flush()
    annotated.flush()
    payload = {"stems": list(cache.stems), "side_px": side_px, "channels": channels}
    (path / _META).write_text(json.dumps(payload), encoding="utf-8")
    for array in (images, masks, positive, annotated, copied_masks):
        if array is None:
            continue
        mapped = getattr(array, "_mmap", None)
        if mapped is not None:
            mapped.close()
    return open_side_pack(path)


@dataclass(frozen=True, slots=True)
class SplitRecipe:
    """Seeded fractions of sites.

    Attributes:
        seed: Shuffle seed.
        train_fraction: Share of sites in train.
        val_fraction: Share of sites in validation.
        test_fraction: Share of sites in test.
    """

    seed: int
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    test_fraction: float = 0.15


@dataclass(frozen=True, slots=True)
class LocationSplit:
    """Every tile of a location follows that location.

    Attributes:
        location_to_split: Split name for each location id.
        stems: Image stems in each split, archive order.
    """

    location_to_split: dict[str, str]
    stems: dict[str, tuple[str, ...]]


def assign_location_splits(index: TileIndex, recipe: SplitRecipe) -> LocationSplit:
    """Assign each location id to train, val, or test.

    Args:
        index: Corpus index.
        recipe: Seed and fractions. Fractions must be positive and sum to 1.

    Returns:
        LocationSplit: Site assignment and the stems that follow it.

    Raises:
        ValueError: If the recipe is invalid or fewer than three sites exist.

    Notes:
        Unique location ids are passed to ``assign_group_splits``. That function
        shuffles groups with ``numpy.random.default_rng(recipe.seed)``. Tiles
        keep archive order inside each split.
    """
    parts = (recipe.train_fraction, recipe.val_fraction, recipe.test_fraction)
    if any(part <= 0.0 for part in parts):
        raise ValueError("split fractions must be > 0")
    if abs(sum(parts) - 1.0) > 1e-6:
        raise ValueError(f"split fractions must sum to 1.0; got {sum(parts)}")
    locations = tuple(dict.fromkeys(tile.location_id for tile in index.tiles))
    if len(locations) < 3:
        raise ValueError(f"need at least 3 locations to split; got {len(locations)}")
    grouped_rows = assign_group_splits(
        locations,
        GroupSplitRecipe(
            seed=recipe.seed,
            train_fraction=recipe.train_fraction,
            val_fraction=recipe.val_fraction,
            test_fraction=recipe.test_fraction,
        ),
    )
    location_to_split: dict[str, str] = {}
    for name in ("train", "val", "test"):
        for row in grouped_rows.for_name(name):
            location_to_split[locations[row]] = name
    grouped: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for tile in index.tiles:
        grouped[location_to_split[tile.location_id]].append(tile.stem)
    stems = {name: tuple(values) for name, values in grouped.items()}
    return LocationSplit(location_to_split=location_to_split, stems=stems)
