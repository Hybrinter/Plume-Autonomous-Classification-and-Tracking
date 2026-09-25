"""Sentinel-2 band identity and subset selection for Zenodo 4250706.

Contains:
  - BandOrder: verified file order of Sentinel-2 ids.
  - BandSpec: named subset request.
  - BandSubset: ids and file indices for one run.
  - coerce_descriptions: fill the corpus order when descriptions are empty.
  - verify_band_order: descriptions to a BandOrder.
  - resolve_subset: a BandSpec against a verified order.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

_CANONICAL: dict[str, str] = {
    "B1": "B1",
    "B01": "B1",
    "B2": "B2",
    "B02": "B2",
    "B3": "B3",
    "B03": "B3",
    "B4": "B4",
    "B04": "B4",
    "B5": "B5",
    "B05": "B5",
    "B6": "B6",
    "B06": "B6",
    "B7": "B7",
    "B07": "B7",
    "B8": "B8",
    "B08": "B8",
    "B8A": "B8A",
    "B08A": "B8A",
    "B9": "B9",
    "B09": "B9",
    "B10": "B10",
    "B11": "B11",
    "B12": "B12",
}

_RGB: tuple[str, ...] = ("B2", "B3", "B4")
_ZENODO_IDS: tuple[str, ...] = (
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B9",
    "B11",
    "B12",
    "B10",
)


@dataclass(frozen=True, slots=True)
class BandOrder:
    """Verified Sentinel-2 id of each GeoTIFF band.

    Attributes:
        ids: File order, one id per band.
        index_by_id: Rasterio-style 0-based index of each id.
    """

    ids: tuple[str, ...]
    index_by_id: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class BandSpec:
    """A named band subset.

    Attributes:
        kind: ``rgb``, ``s2_12``, or ``loo``.
        dropped_band: Sentinel-2 id removed when ``kind`` is ``loo``.
    """

    kind: str
    dropped_band: str = ""


@dataclass(frozen=True, slots=True)
class BandSubset:
    """Channels selected for one run, in file order.

    Attributes:
        name: Stable subset name.
        ids: Sentinel-2 ids in file order.
        indices: 0-based indices into the GeoTIFF stack.
    """

    name: str
    ids: tuple[str, ...]
    indices: tuple[int, ...]


def _token(description: str) -> str:
    """Return the canonical Sentinel-2 id inside one band description."""
    compact = "".join(ch for ch in description.upper() if ch.isalnum())
    for needle, canonical in sorted(_CANONICAL.items(), key=lambda item: -len(item[0])):
        if needle in compact:
            return canonical
    raise ValueError(f"band description {description!r} has no Sentinel-2 id")


def coerce_descriptions(descriptions: Sequence[str]) -> tuple[str, ...]:
    """Fill the Zenodo 4250706 order when every description is empty.

    Args:
        descriptions: One description per GeoTIFF band. Missing text is empty.

    Returns:
        tuple[str, ...]: The input when any description is present, otherwise
        the 13 Sentinel-2 ids used by this corpus, with B10 last.

    Raises:
        ValueError: If every description is empty and the count is not 13.
    """
    cleaned = tuple(item.strip() for item in descriptions)
    if cleaned and all(not item for item in cleaned):
        if len(cleaned) != len(_ZENODO_IDS):
            raise ValueError(
                f"empty descriptions need {len(_ZENODO_IDS)} bands; got {len(cleaned)}"
            )
        return _ZENODO_IDS
    return cleaned


def verify_band_order(descriptions: Sequence[str]) -> BandOrder:
    """Map GeoTIFF band descriptions to Sentinel-2 ids.

    Args:
        descriptions: One description string per file band.

    Returns:
        BandOrder: File order and the index of each id.

    Raises:
        ValueError: If a description has no id, an id repeats, or B10 is absent.
    """
    if not descriptions:
        raise ValueError("band descriptions are empty")
    ids: list[str] = []
    index_by_id: dict[str, int] = {}
    for index, description in enumerate(descriptions):
        if not description or not description.strip():
            raise ValueError(f"band {index} has an empty description")
        band_id = _token(description)
        if band_id in index_by_id:
            raise ValueError(f"duplicate band id {band_id}")
        ids.append(band_id)
        index_by_id[band_id] = index
    if "B10" not in index_by_id:
        raise ValueError("band descriptions do not include B10")
    return BandOrder(ids=tuple(ids), index_by_id=index_by_id)


def _s2_12_ids(order: BandOrder) -> tuple[str, ...]:
    """Return file-order ids with B10 removed."""
    return tuple(band_id for band_id in order.ids if band_id != "B10")


def resolve_subset(order: BandOrder, spec: BandSpec) -> BandSubset:
    """Resolve a subset against a verified band order.

    Args:
        order: Verified file order.
        spec: Subset request.

    Returns:
        BandSubset: Ids and file indices in file order.

    Raises:
        ValueError: If the kind is unknown or a requested id is absent.
    """
    if spec.kind == "rgb":
        selected = _RGB
        name = "rgb"
    elif spec.kind == "s2_12":
        selected = _s2_12_ids(order)
        name = "s2_12"
    elif spec.kind == "loo":
        kept = _s2_12_ids(order)
        if spec.dropped_band not in kept:
            raise ValueError(f"leave-one-out band {spec.dropped_band!r} is not in the 12-band set")
        selected = tuple(band_id for band_id in kept if band_id != spec.dropped_band)
        name = f"loo_{spec.dropped_band}"
    else:
        raise ValueError(f"unknown band set {spec.kind!r}")
    missing = [band_id for band_id in selected if band_id not in order.index_by_id]
    if missing:
        raise ValueError(f"band order is missing {missing}")
    indices = tuple(order.index_by_id[band_id] for band_id in selected)
    return BandSubset(name=name, ids=selected, indices=indices)
