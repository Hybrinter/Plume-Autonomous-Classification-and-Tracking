"""AP-3200T prism weights and the 76 px Sentinel-2 proxy pack.

Contains:
  - PROXY_SIDE_PX, PROXY_GSD_M: 76 px and ``1200 / 76`` metres.
  - WeightTable, load_weight_table: per-color Sentinel-2 weights.
  - mix_prism: L2A counts to BLUE, GREEN, RED reflectance.
  - to_proxy_chip: native mix, area resample to 76, polygon mask at 76.
  - write_prism_pack: annotated tiles only, written with ``write_processed_pack``.

The committed table is curve-height readings of the AP-3200T solid IR-cut
figure. It is not a laboratory integral. B5 and longer bands are absent.
Their weight is 0. This module does not download archives.
"""

from __future__ import annotations

import math
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from tools.ml_models.data.bands import coerce_descriptions, verify_band_order
from tools.ml_models.data.grid import EXTENT_M, rasterize_percent_mask, resample_area
from tools.ml_models.data.meta import DatasetMeta, Provenance
from tools.ml_models.data.pack import write_processed_pack
from tools.ml_models.data.split import SplitRecipe
from tools.ml_models.data.zenodo import TileRef, build_index, iter_stacks, to_native_stack

PROXY_SIDE_PX = 76
PROXY_GSD_M = EXTENT_M / float(PROXY_SIDE_PX)
SOURCE_DOI = "10.5281/zenodo.4250706"
_DN_SCALE = np.float32(10000.0)
_COLOR_NAMES: tuple[str, ...] = ("blue", "green", "red")
_TABLE_KEYS: frozenset[str] = frozenset({"id", "blue", "green", "red"})
_BAND_NAMES: tuple[str, ...] = ("BLUE", "GREEN", "RED")


@dataclass(frozen=True, slots=True)
class WeightTable:
    """Per-color Sentinel-2 weights.

    Attributes:
        id: Table identifier stored on the pack provenance.
        blue: Band id to weight. The weights sum to 1.
        green: Band id to weight. The weights sum to 1.
        red: Band id to weight. The weights sum to 1.
    """

    id: str
    blue: Mapping[str, float]
    green: Mapping[str, float]
    red: Mapping[str, float]

    def color_map(self, name: str) -> Mapping[str, float]:
        """Return the weight map for ``blue``, ``green``, or ``red``.

        Args:
            name: Color name.

        Returns:
            Mapping[str, float]: Band weights for that color.

        Raises:
            ValueError: If ``name`` is not a color.
        """
        if name == "blue":
            return self.blue
        if name == "green":
            return self.green
        if name == "red":
            return self.red
        raise ValueError(f"unknown color {name!r}")


def load_weight_table(path: str | Path) -> WeightTable:
    """Load a prism weight table.

    Args:
        path: TOML file with ``id`` and ``[blue]``, ``[green]``, and ``[red]``
            tables.

    Returns:
        WeightTable: Identifier and one immutable map per color.

    Raises:
        FileNotFoundError: If ``path`` is missing.
        ValueError: If a color is missing or its weights do not sum to 1
            within ``1e-6``.
        tomllib.TOMLDecodeError: If the file is not TOML.
    """
    dest = Path(path)
    if not dest.is_file():
        raise FileNotFoundError(dest)
    raw = tomllib.loads(dest.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("weight table must be a TOML table")
    extra = sorted(set(raw) - _TABLE_KEYS)
    missing = sorted(_TABLE_KEYS - set(raw))
    if extra or missing:
        raise ValueError(f"weight table keys mismatch; missing={missing} extra={extra}")
    table_id = raw["id"]
    if not isinstance(table_id, str) or not table_id:
        raise ValueError("weight table id must be a non-empty string")
    return WeightTable(
        id=table_id,
        blue=MappingProxyType(_color_weights(raw, "blue")),
        green=MappingProxyType(_color_weights(raw, "green")),
        red=MappingProxyType(_color_weights(raw, "red")),
    )


def mix_prism(
    stack_dn: np.ndarray,
    band_ids: Sequence[str],
    table: WeightTable,
) -> np.ndarray:
    """Mix Sentinel-2 L2A counts into BLUE, GREEN, RED reflectance.

    Args:
        stack_dn: Array ``(C, H, W)`` of L2A digital numbers.
        band_ids: Sentinel-2 id of each plane, length ``C``.
        table: Per-color weights. A band absent from a color contributes 0.

    Returns:
        np.ndarray[float32, (3, H, W)]: Planes in order BLUE, GREEN, RED.
        Each plane is the weighted sum of ``clip(stack_dn / 10000, 0, 1)``.

    Raises:
        ValueError: If the stack rank disagrees with ``band_ids``, a band id
            repeats, or the table names a band id that is not in ``band_ids``.
    """
    stack = np.asarray(stack_dn)
    if stack.ndim != 3:
        raise ValueError(f"stack_dn must have shape (C, H, W); got {stack.shape}")
    ids = tuple(band_ids)
    if int(stack.shape[0]) != len(ids):
        raise ValueError(f"stack channels {stack.shape[0]} != len(band_ids) {len(ids)}")
    if int(stack.shape[1]) < 1 or int(stack.shape[2]) < 1:
        raise ValueError(f"stack spatial shape must be positive; got {stack.shape}")
    index_by_id: dict[str, int] = {}
    for index, band_id in enumerate(ids):
        if band_id in index_by_id:
            raise ValueError(f"duplicate band id {band_id}")
        index_by_id[band_id] = index
    for color in _COLOR_NAMES:
        for band_id in table.color_map(color):
            if band_id not in index_by_id:
                raise ValueError(f"unknown band id {band_id}")
    reflectance = np.clip(
        np.asarray(stack, dtype=np.float32) / _DN_SCALE,
        0.0,
        1.0,
    )  # np.ndarray[float32, (C, H, W)]
    mixed = np.zeros((3, int(stack.shape[1]), int(stack.shape[2])), dtype=np.float32)
    for plane, color in enumerate(_COLOR_NAMES):
        for band_id, weight in table.color_map(color).items():
            mixed[plane] += np.float32(weight) * reflectance[index_by_id[band_id]]
    return mixed


def to_proxy_chip(
    stack_dn: np.ndarray,
    band_ids: Sequence[str],
    table: WeightTable,
    polygons: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Mix a native stack and resample the chip and mask to 76 px.

    Args:
        stack_dn: Sentinel-2 L2A counts ``(C, H, W)``.
        band_ids: Sentinel-2 id per plane.
        table: Prism weights.
        polygons: Percent-coordinate ``(V, 2)`` polygons.

    Returns:
        tuple[np.ndarray, np.ndarray]: Image ``(3, 76, 76)`` and mask
        ``(1, 76, 76)``. The image is mixed at native resolution, then
        area-resampled. The mask is rasterized at 76. Ground sample distance
        is ``PROXY_GSD_M`` (``1200 / 76``).

    Raises:
        ValueError: If ``mix_prism`` or the resampler rejects the inputs.
    """
    mixed = mix_prism(stack_dn, band_ids, table)
    image = resample_area(mixed, PROXY_SIDE_PX)
    mask = rasterize_percent_mask(polygons, PROXY_SIDE_PX)
    return image, mask


def write_prism_pack(
    images_tar: str | Path,
    labels_tar: str | Path,
    weights_path: str | Path,
    dest: str | Path,
    *,
    recipe: SplitRecipe | None = None,
) -> DatasetMeta:
    """Write the polygon pack at 76 px.

    Args:
        images_tar: Zenodo image archive. This function does not download it.
        labels_tar: Zenodo label archive.
        weights_path: Prism weight table. A missing file raises before the
            archives are opened.
        dest: Pack directory.
        recipe: Location-group split. The default is seed 0 and fractions
            0.70 / 0.15 / 0.15.

    Returns:
        DatasetMeta: Sidecar written by ``write_processed_pack``.

    Raises:
        FileNotFoundError: If the weight table or an archive is missing.
        ValueError: If no tile is annotated, fewer than three location ids
            are present, or a stack side is more than 2 pixels from 120.

    Notes:
        Rows are tiles whose ``polygons`` value is not ``None``. An empty
        tuple is an annotated negative and stays in the pack. A missing
        annotation is omitted. Each stack is fitted to 120 by 120 before
        ``to_proxy_chip``. A short side is edge-padded. A long side is
        cropped from the origin. The label is 1 when the 76 px mask has a
        positive pixel, otherwise 0. ``group_ids`` are location ids. There is
        no second presence-only pack.
    """
    table = load_weight_table(weights_path)
    index = build_index(Path(images_tar), Path(labels_tar))
    selected = tuple(tile for tile in index.tiles if tile.polygons is not None)
    if not selected:
        raise ValueError("prism pack needs at least one annotated tile")
    images, masks, labels, group_ids = _read_proxy_rows(Path(images_tar), selected, table)
    provenance = Provenance(
        ingest_path="sentinel2_4250706_prism_proxy",
        radiometry="s2_l2a_reflectance",
        gsd_m=PROXY_GSD_M,
        extent_m=EXTENT_M,
        weight_table_id=table.id,
        band_names=_BAND_NAMES,
        norm="unit",
        bit_depth=12,
    )
    return write_processed_pack(
        dest,
        images,
        masks,
        labels,
        provenance,
        SOURCE_DOI,
        group_ids=group_ids,
        recipe=recipe,
    )


def _read_proxy_rows(
    images_tar: Path,
    tiles: Sequence[TileRef],
    table: WeightTable,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    """Return stacked proxy chips for ``tiles``.

    Args:
        images_tar: Image archive.
        tiles: Annotated tiles. ``polygons`` is a tuple. An empty tuple is a
            negative. Each stack is fitted to ``(C, 120, 120)`` before the
            proxy chip.
        table: Prism weights.

    Returns:
        tuple: Images ``(N, 3, 76, 76)``, masks ``(N, 1, 76, 76)``, labels
        ``(N, 1)``, and location ids in archive order.

    Raises:
        ValueError: If a later tile changes the band order, or a stack side
            is more than 2 pixels from 120.
        FileNotFoundError: If a requested member is missing.
    """
    image_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    label_rows: list[float] = []
    group_ids: list[str] = []
    band_ids: tuple[str, ...] | None = None
    for tile, stack, descriptions in iter_stacks(images_tar, tiles):
        ids = verify_band_order(coerce_descriptions(descriptions)).ids
        if band_ids is None:
            band_ids = ids
        elif ids != band_ids:
            raise ValueError(f"band order changed at {tile.stem}")
        polygons = () if tile.polygons is None else tile.polygons
        native = to_native_stack(stack)
        image, mask = to_proxy_chip(native, ids, table, polygons)
        image_rows.append(np.clip(image, 0.0, 1.0).astype(np.float32))
        mask_rows.append(mask)
        label_rows.append(1.0 if bool(np.any(mask > 0.0)) else 0.0)
        group_ids.append(tile.location_id)
    if band_ids is None:
        raise ValueError("prism pack read no tiles")
    images = np.stack(image_rows, axis=0)  # np.ndarray[float32, (N, 3, 76, 76)]
    masks = np.stack(mask_rows, axis=0)  # np.ndarray[float32, (N, 1, 76, 76)]
    labels = np.asarray(label_rows, dtype=np.float32).reshape(-1, 1)
    return images, masks, labels, tuple(group_ids)


def _color_weights(raw: dict[str, object], color: str) -> dict[str, float]:
    """Return finite weights for one color, summing to 1.

    Args:
        raw: Decoded TOML object.
        color: ``blue``, ``green``, or ``red``.

    Returns:
        dict[str, float]: Band id to weight.

    Raises:
        ValueError: If the table is missing, a weight is not finite, or the
            sum is not 1 within ``1e-6``.
    """
    value = raw[color]
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{color} must be a non-empty table of band weights")
    weights: dict[str, float] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{color} band id must be a string")
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{color} weight for {key} must be a number")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{color} weight for {key} must be finite")
        weights[key] = number
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"{color} weights sum to {total}; expected 1 within 1e-6")
    return weights
