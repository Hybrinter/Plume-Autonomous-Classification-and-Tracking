"""Tile index over the Zenodo image and label archives.

Contains:
  - TileRef: one image stem.
  - TileIndex: the corpus index.
  - build_index: stems, presence, and polygons from two tar archives.
  - read_stack: one GeoTIFF member as a float stack plus band descriptions.
"""

from __future__ import annotations

import io
import json
import tarfile
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


def read_stack(images_tar: Path, ref: TileRef) -> tuple[np.ndarray, tuple[str, ...]]:
    """Read one GeoTIFF member.

    Args:
        images_tar: Image archive.
        ref: Tile whose ``member_name`` is read.

    Returns:
        tuple: Float32 array ``(C, H, W)`` and one description per band.
        A missing description is an empty string.

    Raises:
        FileNotFoundError: If the member is absent.
        ImportError: If rasterio is not installed.
    """
    import rasterio

    with tarfile.open(images_tar, "r:*") as archive:
        extracted = archive.extractfile(ref.member_name)
        if extracted is None:
            raise FileNotFoundError(ref.member_name)
        payload = extracted.read()
    with rasterio.open(io.BytesIO(payload)) as dataset:
        stack = dataset.read().astype(np.float32)
        descriptions = tuple("" if item is None else str(item) for item in dataset.descriptions)
    return stack, descriptions
