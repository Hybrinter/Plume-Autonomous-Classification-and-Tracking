"""Tile index over the Zenodo image and label archives.

Contains:
  - TileRef: one image stem.
  - TileIndex: the corpus index.
  - build_index: stems, presence, and polygons from two tar archives.
  - iter_stacks: one forward pass over requested GeoTIFF members.
"""

from __future__ import annotations

import io
import json
import tarfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_GEOTIFF_SUFFIXES = (".tif", ".tiff")
_LABEL_SUFFIX = "_features.json"


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
