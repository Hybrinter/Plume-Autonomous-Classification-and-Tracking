"""Zenodo 4250706 archive index and forward GeoTIFF reads.

Contains:
  - TileRef, TileIndex, location_id_of, build_index.
  - iter_stacks, to_native_stack.

Image members must sit under exactly one of a ``positive`` or ``negative``
class directory. Label polygons come from ``annotations.parse_polygons``.
Rasterio is imported inside the GeoTIFF reader. Nothing extracts to disk.
"""

from __future__ import annotations

import io
import json
import tarfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np

from tools.ml_models.dataset.sources.zenodo.annotations import (
    parse_polygons as _parse_polygons,
)
from tools.ml_models.dataset.sources.zenodo.bins import NATIVE_SIDE

_GEOTIFF_SUFFIXES = (".tif", ".tiff")
_LABEL_SUFFIX = "_features.json"
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
        ValueError: If an image stem has no location token, or an image member
            path is not under exactly one of ``positive`` or ``negative``.
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
            parts = PurePosixPath(member.name).parts
            positive = "positive" in parts
            if positive == ("negative" in parts):
                raise ValueError(
                    f"image member {member.name!r} must sit under exactly one of "
                    "'positive' or 'negative'"
                )
            stem = Path(filename).stem
            if stem in seen:
                continue
            seen.add(stem)
            location = location_id_of(stem)
            if not location:
                raise ValueError(f"stem {stem!r} has an empty location id")
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
