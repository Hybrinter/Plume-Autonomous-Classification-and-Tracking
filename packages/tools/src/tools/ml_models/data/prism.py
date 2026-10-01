"""AP-3200T prism weights, the 76 px chip pack, and stored flight tiles.

Contains:
  - PROXY_SIDE_PX, PROXY_GSD_M: 76 px and ``1200 / 76`` metres.
  - CHIP_HW, TILE_HW, FRAME_HW, TILE_GRID, GSD_MATCH_TOLERANCE: chip
    ``(76, 76)``, tile ``(193, 258)``, frame ``(1544, 2064)``, an ``8`` by
    ``8`` grid, and a 1 percent ground-sample-distance tolerance.
  - WeightTable, load_weight_table: per-color Sentinel-2 weights.
  - mix_prism: L2A counts to BLUE, GREEN, RED reflectance.
  - to_proxy_chip: native mix, area resample to 76, polygon mask at 76.
  - write_prism_pack: annotated chips, written with ``write_processed_pack``.
  - write_tile_pack: one stored tile per source chip at ``TILE_HW``.
  - LocationSplit, union_location_split: one location split for two packs.

``TILE_HW`` is ``(along-track, lateral)``. ``8 * 193 = 1544`` and
``8 * 258 = 2064``, so the tensor layout is ``(N, C, 193, 258)``. The
committed table is curve-height readings of the AP-3200T solid IR-cut
figure. It is not a laboratory integral. B5 and longer bands are absent.
Their weight is 0. This module does not download archives.
``union_location_split`` does not call ``concat_packs``.
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
from tools.ml_models.data.meta import DatasetMeta, IngestPath, Provenance
from tools.ml_models.data.pack import ProcessedPack, load_processed_pack, write_processed_pack
from tools.ml_models.data.split import SplitIndex, SplitRecipe, assign_group_splits
from tools.ml_models.data.zenodo import TileRef, build_index, iter_stacks, to_native_stack

PROXY_SIDE_PX = 76
PROXY_GSD_M = EXTENT_M / float(PROXY_SIDE_PX)
CHIP_HW: tuple[int, int] = (PROXY_SIDE_PX, PROXY_SIDE_PX)
TILE_HW: tuple[int, int] = (193, 258)
FRAME_HW: tuple[int, int] = (1544, 2064)
TILE_GRID: tuple[int, int] = (8, 8)
GSD_MATCH_TOLERANCE: float = 0.01
_TILE_INGEST: IngestPath = "sentinel2_4250706_prism_tile"
_TILE_FEATHER_PX = 6
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


@dataclass(frozen=True, slots=True)
class LocationSplit:
    """One location split for the union of two packs' group ids.

    Attributes:
        groups: Unique group ids. Left-pack ids come first, in first-seen
            order. Ids that appear only on the right pack follow.
        names: ``train``, ``val``, or ``test`` for each entry of ``groups``.
        left: Row indices into the left pack. Rows keep input order inside
            each split.
        right: Row indices into the right pack. Rows keep input order inside
            each split.
    """

    groups: tuple[str, ...]
    names: tuple[str, ...]
    left: SplitIndex
    right: SplitIndex


def write_tile_pack(
    source: str | Path | ProcessedPack,
    dest: str | Path,
    *,
    seed: int = 0,
    recipe: SplitRecipe | None = None,
) -> DatasetMeta:
    """Write one stored flight tile per chip in ``source``.

    Args:
        source: Chip pack directory, or an in-memory pack. A directory is
            loaded with ``load_processed_pack``.
        dest: Tile pack directory.
        seed: Seed for ``numpy.random.default_rng``. The generator drives the
            mosaic phase and the paste offset.
        recipe: Location-group split for the tile rows. The default is seed 0
            and fractions 0.70 / 0.15 / 0.15.

    Returns:
        DatasetMeta: Sidecar written by ``write_processed_pack``.

    Raises:
        FileNotFoundError: If a pack directory is missing a file.
        ValueError: If ``seed`` is not an int, the source is not a 76 px
            BLUE/GREEN/RED unit-reflectance pack, its ground sample distance
            is outside ``GSD_MATCH_TOLERANCE`` of ``PROXY_GSD_M``, group ids
            are missing, or a split has a positive chip and no negative chip.

    Notes:
        Each source row becomes one tile of ``TILE_HW`` ``(193, 258)``. A row
        with label 0 is a fill of same-split negative chips. A row with
        label above 0 places that annotated chip on the fill. The offset
        keeps one positive mask pixel inside the tile. The image border of
        6 pixels blends into the tile. The mask keeps the
        polygon pixels that land in the tile. Group ids are the source
        location ids, in source order. The tile provenance uses ingest
        path ``sentinel2_4250706_prism_tile``, ground sample distance
        ``PROXY_GSD_M``, bands BLUE, GREEN, RED, norm ``unit``, and radiometry
        ``s2_l2a_reflectance``. ``extent_m`` stays 1200 m.
    """
    _require_tile_geometry()
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed must be an int; got {seed!r}")
    pack = source if isinstance(source, ProcessedPack) else load_processed_pack(source)
    _require_chip_pack(pack)
    group_ids = pack.group_ids
    if group_ids is None:
        raise ValueError("tile source needs location group ids")
    images, masks, labels = _build_tile_arrays(pack, group_ids, seed)
    provenance = Provenance(
        ingest_path=_TILE_INGEST,
        radiometry="s2_l2a_reflectance",
        gsd_m=PROXY_GSD_M,
        extent_m=EXTENT_M,
        weight_table_id=pack.meta.weight_table_id,
        band_names=_BAND_NAMES,
        norm="unit",
        bit_depth=pack.meta.bit_depth,
    )
    return write_processed_pack(
        dest,
        images,
        masks,
        labels,
        provenance,
        pack.meta.source_doi,
        group_ids=group_ids,
        recipe=recipe,
    )


def union_location_split(
    left: ProcessedPack,
    right: ProcessedPack,
    *,
    recipe: SplitRecipe | None = None,
    gsd_tolerance: float = GSD_MATCH_TOLERANCE,
) -> LocationSplit:
    """Assign one location split to the union of two packs' group ids.

    Args:
        left: First pack. Group ids are required.
        right: Second pack. Group ids are required.
        recipe: Seed and fractions for ``assign_group_splits``. The default
            is seed 0 and fractions 0.70 / 0.15 / 0.15.
        gsd_tolerance: Maximum relative ground-sample-distance gap. The
            default is ``GSD_MATCH_TOLERANCE`` (0.01).

    Returns:
        LocationSplit: Unique groups, the split name of each group, and row
        indices for each pack.

    Raises:
        ValueError: If band names differ, norm differs, ground sample distance
            differs by more than ``gsd_tolerance``, either pack lacks group
            ids, or fewer than three unique groups are present.

    Notes:
        A match requires ``abs(left - right) / max(left, right) <= gsd_tolerance``.
        Both distances must be finite and greater than 0. Spatial size is not
        compared. A group id may appear in both packs. This function does not
        call ``concat_packs``.
    """
    _require_gsd_tolerance(gsd_tolerance)
    _require_shared_contract(left, right, gsd_tolerance)
    left_ids = left.group_ids
    right_ids = right.group_ids
    if left_ids is None or right_ids is None:
        raise ValueError("union split needs group ids on both packs")
    if len(left_ids) != left.meta.n or len(right_ids) != right.meta.n:
        raise ValueError("len(group_ids) must equal N on both packs")
    groups = tuple(dict.fromkeys((*left_ids, *right_ids)))
    assigned = assign_group_splits(groups, recipe if recipe is not None else SplitRecipe())
    group_to_name: dict[str, str] = {}
    for split_name in ("train", "val", "test"):
        for row in assigned.for_name(split_name):
            group_to_name[groups[row]] = split_name
    names = tuple(group_to_name[group_id] for group_id in groups)
    return LocationSplit(
        groups=groups,
        names=names,
        left=_project_split(left_ids, group_to_name),
        right=_project_split(right_ids, group_to_name),
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


def _require_tile_geometry() -> None:
    """Raise ValueError when the tile grid does not cover the flight frame.

    Returns:
        None.

    Raises:
        ValueError: If ``TILE_GRID`` times ``TILE_HW`` is not ``FRAME_HW``, or
            ``CHIP_HW`` is not the 76 px proxy.
    """
    grid_y, grid_x = TILE_GRID
    tile_h, tile_w = TILE_HW
    frame_h, frame_w = FRAME_HW
    if grid_y * tile_h != frame_h or grid_x * tile_w != frame_w:
        raise ValueError(f"tile grid {TILE_GRID} times tile {TILE_HW} must equal frame {FRAME_HW}")
    if CHIP_HW != (PROXY_SIDE_PX, PROXY_SIDE_PX):
        raise ValueError(f"chip size {CHIP_HW} must equal {(PROXY_SIDE_PX, PROXY_SIDE_PX)}")


def _require_gsd_tolerance(tolerance: float) -> None:
    """Raise ValueError when ``tolerance`` is not a finite number >= 0.

    Args:
        tolerance: Relative ground-sample-distance gap.

    Returns:
        None.

    Raises:
        ValueError: If ``tolerance`` is a bool, not a number, non-finite, or
            negative.
    """
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
        raise ValueError(f"gsd_tolerance must be a number; got {tolerance!r}")
    if not math.isfinite(float(tolerance)) or float(tolerance) < 0.0:
        raise ValueError(f"gsd_tolerance must be finite and >= 0; got {tolerance!r}")


def _gsd_within(left: float, right: float, tolerance: float) -> bool:
    """Return True when two distances match within ``tolerance``.

    Args:
        left: Ground sample distance in metres.
        right: Ground sample distance in metres.
        tolerance: Relative gap. ``abs(left - right) / max(left, right)``.

    Returns:
        bool: True when both distances are finite and greater than 0 and the
        relative gap is at most ``tolerance``.
    """
    if not math.isfinite(left) or not math.isfinite(right) or left <= 0.0 or right <= 0.0:
        return False
    scale = max(left, right)
    return abs(left - right) / scale <= float(tolerance)


def _require_chip_pack(pack: ProcessedPack) -> None:
    """Raise ValueError when ``pack`` is not a prism chip pack.

    Args:
        pack: Candidate chip pack.

    Returns:
        None.

    Raises:
        ValueError: If the spatial size, bands, norm, radiometry, ground
            sample distance, group ids, or split coverage disagree with the
            chip contract.
    """
    meta = pack.meta
    if meta.height != CHIP_HW[0] or meta.width != CHIP_HW[1]:
        raise ValueError(
            f"chip pack spatial size must be {CHIP_HW[0]}x{CHIP_HW[1]}; "
            f"got {meta.height}x{meta.width}"
        )
    if meta.band_names != _BAND_NAMES:
        raise ValueError(f"chip pack bands must be {_BAND_NAMES}; got {meta.band_names}")
    if meta.norm != "unit":
        raise ValueError(f"chip pack norm must be 'unit'; got {meta.norm!r}")
    if meta.radiometry != "s2_l2a_reflectance":
        raise ValueError(
            f"chip pack radiometry must be 's2_l2a_reflectance'; got {meta.radiometry!r}"
        )
    if not _gsd_within(meta.gsd_m, PROXY_GSD_M, GSD_MATCH_TOLERANCE):
        raise ValueError(
            f"chip pack gsd_m {meta.gsd_m} is outside {GSD_MATCH_TOLERANCE} of {PROXY_GSD_M}"
        )
    if pack.group_ids is None:
        raise ValueError("tile source needs location group ids")
    if len(pack.group_ids) != meta.n:
        raise ValueError(f"len(group_ids) must equal N; got {len(pack.group_ids)} and {meta.n}")
    _row_splits(pack)


def _require_shared_contract(left: ProcessedPack, right: ProcessedPack, tolerance: float) -> None:
    """Raise ValueError when packs do not share bands, norm, and distance.

    Args:
        left: First pack.
        right: Second pack.
        tolerance: Relative ground-sample-distance gap.

    Returns:
        None.

    Raises:
        ValueError: If band names differ, norm differs, or the ground sample
            distance gap exceeds ``tolerance``.
    """
    if left.meta.band_names != right.meta.band_names:
        raise ValueError(
            f"band_names mismatch: {left.meta.band_names!r} != {right.meta.band_names!r}"
        )
    if left.meta.norm != right.meta.norm:
        raise ValueError(f"norm mismatch: {left.meta.norm!r} != {right.meta.norm!r}")
    if not _gsd_within(left.meta.gsd_m, right.meta.gsd_m, tolerance):
        raise ValueError(
            f"gsd_m mismatch: {left.meta.gsd_m} and {right.meta.gsd_m} "
            f"differ by more than {float(tolerance)}"
        )


def _row_splits(pack: ProcessedPack) -> tuple[str, ...]:
    """Return the split name of each row.

    Args:
        pack: Pack whose ``splits`` cover ``0..N-1``.

    Returns:
        tuple[str, ...]: ``train``, ``val``, or ``test`` per row.

    Raises:
        ValueError: If an index is not an int, is out of range, is duplicated,
            or a row has no split.
    """
    n = pack.meta.n
    names: list[str] = [""] * n
    seen = 0
    for split_name in ("train", "val", "test"):
        for row in pack.splits.for_name(split_name):
            if isinstance(row, bool) or not isinstance(row, int):
                raise ValueError(f"{split_name} indices must be integers")
            if row < 0 or row >= n:
                raise ValueError(f"split index {row} out of range for n={n}")
            if names[row]:
                raise ValueError(f"duplicate index {row} in splits")
            names[row] = split_name
            seen += 1
    if seen != n:
        raise ValueError(f"splits must cover 0..{n - 1}")
    return tuple(names)


def _overlap_window(
    frame_hw: tuple[int, int],
    chip_hw: tuple[int, int],
    top: int,
    left: int,
) -> tuple[int, int, int, int, int, int] | None:
    """Return the overlapping source and destination rectangles.

    Args:
        frame_hw: Destination ``(height, width)``.
        chip_hw: Chip ``(height, width)``.
        top: Destination row of the chip origin. Negative values clip the chip.
        left: Destination column of the chip origin.

    Returns:
        tuple[int, int, int, int, int, int] | None: ``src_y``, ``src_x``,
        ``dst_y``, ``dst_x``, ``height``, ``width``. ``None`` when the chip
        misses the frame.
    """
    frame_h, frame_w = frame_hw
    chip_h, chip_w = chip_hw
    src_y = max(0, -top)
    src_x = max(0, -left)
    dst_y = max(0, top)
    dst_x = max(0, left)
    height = min(chip_h - src_y, frame_h - dst_y)
    width = min(chip_w - src_x, frame_w - dst_x)
    if height <= 0 or width <= 0:
        return None
    return src_y, src_x, dst_y, dst_x, height, width


def _feather_alpha(chip_h: int, chip_w: int, feather_px: int) -> np.ndarray:
    """Return per-pixel blend weights, shape ``(chip_h, chip_w)``.

    Args:
        chip_h: Chip height.
        chip_w: Chip width.
        feather_px: Border thickness in pixels. ``0`` is a hard replace.

    Returns:
        np.ndarray[float32, (chip_h, chip_w)]: ``1`` on the interior. On the
        outer ``feather_px`` pixels, the weight is ``(inset + 1) / (feather_px + 1)``.
        ``inset`` is the distance in pixels to the nearest chip edge.
    """
    if feather_px <= 0:
        return np.ones((chip_h, chip_w), dtype=np.float32)
    rows = np.arange(chip_h, dtype=np.float32)
    cols = np.arange(chip_w, dtype=np.float32)
    dist_y = np.minimum(rows, np.float32(chip_h - 1) - rows)
    dist_x = np.minimum(cols, np.float32(chip_w - 1) - cols)
    dist = np.minimum(dist_y[:, None], dist_x[None, :])  # np.ndarray[float32, (H, W)]
    alpha = np.ones((chip_h, chip_w), dtype=np.float32)
    border = dist < float(feather_px)
    alpha[border] = (dist[border] + 1.0) / float(feather_px + 1)
    return alpha


def _feather_paste(
    frame: np.ndarray,
    chip: np.ndarray,
    top: int,
    left: int,
    feather_px: int,
) -> None:
    """Blend ``chip`` onto ``frame`` in place.

    Args:
        frame: Array ``(C, H, W)`` updated in place.
        chip: Array ``(C, h, w)``. The interior replaces the frame. The outer
            ``feather_px`` pixels blend ``alpha * chip + (1 - alpha) * frame``.
        top: Destination row of the chip origin. May be negative.
        left: Destination column of the chip origin. May be negative.
        feather_px: Border thickness. ``0`` replaces every overlapping pixel.

    Returns:
        None.

    Raises:
        ValueError: If the arrays are not ``(C, H, W)``, the channel counts
            differ, or ``feather_px`` is negative.
    """
    if isinstance(feather_px, bool) or not isinstance(feather_px, int) or feather_px < 0:
        raise ValueError(f"feather_px must be an int >= 0; got {feather_px!r}")
    if frame.ndim != 3 or chip.ndim != 3:
        raise ValueError(
            f"frame and chip must have shape (C, H, W); got {frame.shape}, {chip.shape}"
        )
    if int(frame.shape[0]) != int(chip.shape[0]) or int(frame.shape[0]) < 1:
        raise ValueError(f"channel count {frame.shape[0]} != {chip.shape[0]}")
    chip_h = int(chip.shape[1])
    chip_w = int(chip.shape[2])
    if chip_h < 1 or chip_w < 1 or int(frame.shape[1]) < 1 or int(frame.shape[2]) < 1:
        raise ValueError("frame and chip spatial axes must be positive")
    window = _overlap_window(
        (int(frame.shape[1]), int(frame.shape[2])),
        (chip_h, chip_w),
        top,
        left,
    )
    if window is None:
        return
    src_y, src_x, dst_y, dst_x, height, width = window
    alpha = _feather_alpha(chip_h, chip_w, feather_px)
    weights = alpha[src_y : src_y + height, src_x : src_x + width]  # np.ndarray[float32, (h, w)]
    dst = frame[:, dst_y : dst_y + height, dst_x : dst_x + width]
    src = np.asarray(chip[:, src_y : src_y + height, src_x : src_x + width], dtype=np.float32)
    blended = weights[None, :, :] * src + (1.0 - weights[None, :, :]) * dst
    dst[...] = blended


def _paste_mask(frame_mask: np.ndarray, chip_mask: np.ndarray, top: int, left: int) -> None:
    """Write polygon pixels that land inside the frame.

    Args:
        frame_mask: Scene mask ``(1, H, W)``, updated in place.
        chip_mask: Polygon mask ``(1, h, w)``. Values are not blended.
        top: Destination row of the chip origin.
        left: Destination column of the chip origin.

    Returns:
        None.
    """
    window = _overlap_window(
        (int(frame_mask.shape[1]), int(frame_mask.shape[2])),
        (int(chip_mask.shape[1]), int(chip_mask.shape[2])),
        top,
        left,
    )
    if window is None:
        return
    src_y, src_x, dst_y, dst_x, height, width = window
    dst = frame_mask[:, dst_y : dst_y + height, dst_x : dst_x + width]
    src = chip_mask[:, src_y : src_y + height, src_x : src_x + width]
    np.maximum(dst, src, out=dst)


def _offset_keeps_positive(
    chip_mask: np.ndarray,
    frame_h: int,
    frame_w: int,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Return a clipped offset that keeps one positive mask pixel in frame.

    Args:
        chip_mask: Polygon mask ``(1, h, w)`` with at least one value above 0.
        frame_h: Frame height.
        frame_w: Frame width.
        rng: Generator for candidate offsets.

    Returns:
        tuple[int, int]: ``(top, left)`` of the chip origin. Candidates are
        uniform on offsets where the chip rectangle meets the frame. The
        result is the first candidate whose overlap contains a positive pixel.
    """
    chip_h = int(chip_mask.shape[1])
    chip_w = int(chip_mask.shape[2])
    while True:
        top = int(rng.integers(-chip_h + 1, frame_h))
        left = int(rng.integers(-chip_w + 1, frame_w))
        window = _overlap_window((frame_h, frame_w), (chip_h, chip_w), top, left)
        if window is None:
            continue
        src_y, src_x, _dst_y, _dst_x, height, width = window
        overlap = chip_mask[0, src_y : src_y + height, src_x : src_x + width]
        if bool(np.any(overlap > 0.0)):
            return top, left


def _mosaic(
    chips: Sequence[np.ndarray],
    frame_h: int,
    frame_w: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Tile background chips with a random phase.

    Args:
        chips: Label-0 images ``(C, H, W)`` that share a spatial size.
        frame_h: Frame height.
        frame_w: Frame width.
        rng: Generator for the phase and the chip choice at each tile.

    Returns:
        np.ndarray[float32, (C, frame_h, frame_w)]: Hard-copy fill. Tiles
        that cross the frame border are clipped.
    """
    probe = chips[0]
    channels = int(probe.shape[0])
    chip_h = int(probe.shape[1])
    chip_w = int(probe.shape[2])
    frame = np.zeros((channels, frame_h, frame_w), dtype=np.float32)
    phase_y = int(rng.integers(0, chip_h))
    phase_x = int(rng.integers(0, chip_w))
    y = -phase_y
    while y < frame_h:
        x = -phase_x
        while x < frame_w:
            choice = chips[int(rng.integers(0, len(chips)))]
            window = _overlap_window((frame_h, frame_w), (chip_h, chip_w), y, x)
            if window is not None:
                src_y, src_x, dst_y, dst_x, height, width = window
                frame[:, dst_y : dst_y + height, dst_x : dst_x + width] = choice[
                    :, src_y : src_y + height, src_x : src_x + width
                ]
            x += chip_w
        y += chip_h
    return frame


def _compose_frame(
    background: Sequence[np.ndarray],
    frame_hw: tuple[int, int],
    rng: np.random.Generator,
    *,
    plume_image: np.ndarray | None = None,
    plume_mask: np.ndarray | None = None,
    feather_px: int = _TILE_FEATHER_PX,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fill a frame from negative chips and an optional plume chip.

    Args:
        background: Label-0 images ``(C, H, W)`` that share one shape.
        frame_hw: Frame ``(height, width)``.
        rng: Generator for the fill phase and the plume offset.
        plume_image: Annotated chip ``(C, h, w)``. ``None`` leaves the mask empty.
        plume_mask: Polygon mask ``(1, h, w)`` for ``plume_image``.
        feather_px: Image border blend. The mask is not blended.

    Returns:
        tuple[np.ndarray, np.ndarray, float]: Image ``(C, frame_h, frame_w)``,
        mask ``(1, frame_h, frame_w)``, and label. The label is ``1`` when any
        mask pixel is positive.

    Raises:
        ValueError: If ``background`` is empty, the chips disagree on shape,
            the plume mask is missing, or the plume mask has no positive pixel.
    """
    if not background:
        raise ValueError("frame needs a negative chip")
    frame_h, frame_w = frame_hw
    if (
        isinstance(frame_h, bool)
        or isinstance(frame_w, bool)
        or not isinstance(frame_h, int)
        or not isinstance(frame_w, int)
        or frame_h < 1
        or frame_w < 1
    ):
        raise ValueError(f"frame_hw must be positive ints; got {frame_hw!r}")
    probe = np.asarray(background[0], dtype=np.float32)
    if probe.ndim != 3 or int(probe.shape[1]) < 1 or int(probe.shape[2]) < 1:
        raise ValueError(f"background chip must have shape (C, H, W); got {probe.shape}")
    chips: list[np.ndarray] = []
    for chip in background:
        image = np.asarray(chip, dtype=np.float32)
        if image.shape != probe.shape:
            raise ValueError(f"background chip shape {image.shape} != {probe.shape}")
        chips.append(image)
    image = _mosaic(chips, frame_h, frame_w, rng)
    mask = np.zeros((1, frame_h, frame_w), dtype=np.float32)
    if plume_image is None:
        return image, mask, 0.0
    if plume_mask is None:
        raise ValueError("plume chip needs a mask")
    plume = np.asarray(plume_image, dtype=np.float32)
    polygon = np.asarray(plume_mask, dtype=np.float32)
    if plume.ndim != 3 or int(plume.shape[0]) != int(probe.shape[0]):
        raise ValueError(f"plume chip must have shape {probe.shape}; got {plume.shape}")
    if polygon.shape != (1, int(plume.shape[1]), int(plume.shape[2])):
        raise ValueError(
            f"plume mask must have shape {(1, int(plume.shape[1]), int(plume.shape[2]))}; "
            f"got {polygon.shape}"
        )
    if not bool(np.any(polygon > 0.0)):
        raise ValueError("plume mask has no positive pixel")
    top, left = _offset_keeps_positive(polygon, frame_h, frame_w, rng)
    _feather_paste(image, plume, top, left, feather_px)
    _paste_mask(mask, polygon, top, left)
    label = 1.0 if bool(np.any(mask > 0.0)) else 0.0
    return image, mask, label


def _build_tile_arrays(
    pack: ProcessedPack,
    group_ids: tuple[str, ...],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Paste one tile per source chip.

    Args:
        pack: Chip pack that already passed :func:`_require_chip_pack`.
        group_ids: Location id per row.
        seed: Generator seed.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Images
        ``(N, 3, 193, 258)``, masks ``(N, 1, 193, 258)``, and labels ``(N, 1)``.

    Raises:
        ValueError: If a positive row has an empty mask, or its split has no
            negative chip.
    """
    splits = _row_splits(pack)
    negatives: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    rows: list[tuple[np.ndarray, np.ndarray, float, str, str]] = []
    for row, group_id in enumerate(group_ids):
        image = np.array(pack.images[row], dtype=np.float32, copy=True)
        mask = np.array(pack.masks[row], dtype=np.float32, copy=True)
        label = float(pack.labels[row, 0])
        split_name = splits[row]
        rows.append((image, mask, label, split_name, group_id))
        if label <= 0.0:
            negatives[split_name].append(image)
    rng = np.random.default_rng(seed)
    image_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    label_rows: list[float] = []
    for image, mask, label, split_name, group_id in rows:
        background = negatives[split_name]
        if not background:
            raise ValueError(f"split {split_name!r} has no negative chip")
        if label > 0.0:
            if not bool(np.any(mask > 0.0)):
                raise ValueError(f"positive chip {group_id!r} has an empty mask")
            filled, filled_mask, filled_label = _compose_frame(
                background,
                TILE_HW,
                rng,
                plume_image=image,
                plume_mask=mask,
            )
        else:
            filled, filled_mask, filled_label = _compose_frame(background, TILE_HW, rng)
        image_rows.append(np.clip(filled, 0.0, 1.0).astype(np.float32, copy=False))
        mask_rows.append(np.asarray(filled_mask, dtype=np.float32))
        label_rows.append(filled_label)
    images = np.stack(image_rows, axis=0)  # np.ndarray[float32, (N, 3, 193, 258)]
    masks = np.stack(mask_rows, axis=0)  # np.ndarray[float32, (N, 1, 193, 258)]
    labels = np.asarray(label_rows, dtype=np.float32).reshape(-1, 1)
    return images, masks, labels


def _project_split(group_ids: Sequence[str], group_to_name: Mapping[str, str]) -> SplitIndex:
    """Map a group assignment onto pack rows.

    Args:
        group_ids: Group id per row.
        group_to_name: Group id to ``train``, ``val``, or ``test``.

    Returns:
        SplitIndex: Row indices in input order inside each split.
    """
    grouped: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for row, group_id in enumerate(group_ids):
        grouped[group_to_name[group_id]].append(row)
    return SplitIndex(
        train=tuple(grouped["train"]),
        val=tuple(grouped["val"]),
        test=tuple(grouped["test"]),
    )


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
