"""Labeled flight tile directory.

Contains:
  - FlightTileWrite: one tile accepted by the writer.
  - write_flight_tile_dir: create ``source.json``, ``index.jsonl``, ``tiles/``,
    and ``masks/``.
  - FlightTileDir: ``RawSource`` reader.

On-disk images are uint16 ``(3, 193, 258)`` in ``input_bands`` order. Masks are
uint8 ``(193, 258)``. A missing ``group_id`` in ``index.jsonl`` defaults to
``frame_id``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tools.ml_models.dataset.geometry import INPUT_BANDS, tile_hw
from tools.ml_models.dataset.raw import BinSpec, GsdPair, RawTile, RawTileRef

_SOURCE_KEYS: tuple[str, ...] = ("band_names", "bit_depth", "source_ref", "gsd_reference_m")
_INDEX_REQUIRED: tuple[str, ...] = (
    "tile_id",
    "frame_id",
    "row",
    "col",
    "label",
    "has_mask",
    "theta_g_deg",
    "gsd_lateral_m",
    "gsd_along_m",
)
_INDEX_OPTIONAL: tuple[str, ...] = ("group_id", "gsd_nominal")
_ELEVATION_BINS: tuple[int, ...] = (5, 15, 25, 35, 45)


@dataclass(frozen=True, slots=True)
class FlightTileWrite:
    """One tile to store in a flight tile directory.

    Attributes:
        tile_id: File stem.
        frame_id: Frame that contains the tile.
        row: Grid row.
        col: Grid column.
        label: Classification target.
        theta_g_deg: Gimbal elevation at the shutter, in degrees.
        gsd: Pixel GSD at the tile center.
        image: np.ndarray[uint16, (3, 193, 258)].
        mask: np.ndarray[uint8, (193, 258)] or ``(1, 193, 258)``. None when
            the tile has no mask.
        group_id: Split group. None stores ``frame_id``.
        gsd_nominal: True when ``gsd`` is nominal orbit geometry rather than
            measured capture geometry.
    """

    tile_id: str
    frame_id: str
    row: int
    col: int
    label: float
    theta_g_deg: float
    gsd: GsdPair
    image: np.ndarray
    mask: np.ndarray | None = None
    group_id: str | None = None
    gsd_nominal: bool = False


def write_flight_tile_dir(
    dest: str | Path,
    tiles: tuple[FlightTileWrite, ...] | list[FlightTileWrite],
    *,
    band_names: tuple[str, ...] = INPUT_BANDS,
    bit_depth: int = 12,
    source_ref: str = "",
    gsd_reference_m: float = 15.87,
) -> None:
    """Write a flight tile directory.

    Args:
        dest: New directory.
        tiles: Tiles in index order.
        band_names: Channel names stored in ``source.json``.
        bit_depth: ADC depth stored in ``source.json``.
        source_ref: Provenance string.
        gsd_reference_m: Reference metres stored in ``source.json``.

    Returns:
        None.

    Raises:
        FileExistsError: If ``dest`` already exists.
        ValueError: If a tile image, mask, id, or GSD is invalid.
    """
    root = Path(dest)
    if root.exists():
        raise FileExistsError(f"flight tile directory exists: {root}")
    if bit_depth < 1:
        raise ValueError(f"bit_depth must be >= 1; got {bit_depth}")
    if not math.isfinite(gsd_reference_m) or gsd_reference_m <= 0.0:
        raise ValueError("gsd_reference_m must be finite and > 0")
    if len(tiles) < 1:
        raise ValueError("flight tile directory requires at least one tile")
    root.mkdir()
    (root / "tiles").mkdir()
    (root / "masks").mkdir()
    source_payload = {
        "band_names": list(band_names),
        "bit_depth": bit_depth,
        "source_ref": source_ref,
        "gsd_reference_m": gsd_reference_m,
    }
    (root / "source.json").write_text(
        json.dumps(source_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    lines: list[str] = []
    seen: set[str] = set()
    height, width = tile_hw()
    for tile in tiles:
        _validate_write(tile, height, width, seen)
        seen.add(tile.tile_id)
        group_id = tile.frame_id if tile.group_id is None else tile.group_id
        np.save(root / "tiles" / f"{tile.tile_id}.npy", tile.image)
        has_mask = tile.mask is not None
        if tile.mask is not None:
            np.save(root / "masks" / f"{tile.tile_id}.npy", _mask_hw(tile.mask, height, width))
        lines.append(
            json.dumps(
                {
                    "tile_id": tile.tile_id,
                    "frame_id": tile.frame_id,
                    "row": tile.row,
                    "col": tile.col,
                    "group_id": group_id,
                    "label": tile.label,
                    "has_mask": has_mask,
                    "theta_g_deg": tile.theta_g_deg,
                    "gsd_lateral_m": tile.gsd.lateral_m,
                    "gsd_along_m": tile.gsd.along_m,
                    "gsd_nominal": tile.gsd_nominal,
                },
                separators=(",", ":"),
            )
        )
    (root / "index.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


class FlightTileDir:
    """Read a labeled flight tile directory as a raw source.

    Attributes:
        name: Always ``flight``.
        band_names: From ``source.json``.
        domain: Always ``dn``.
        bit_depth: From ``source.json``.
        source_ref: From ``source.json``.
        extent_m: Always None. Tiles are the flight size.
        bins: Always empty.
        gsd_reference_m: From ``source.json``.
    """

    name = "flight"
    domain = "dn"
    extent_m: tuple[float, float] | None = None
    bins: tuple[BinSpec, ...] = ()

    def __init__(self, root: str | Path) -> None:
        """Load ``source.json`` and ``index.jsonl``.

        Args:
            root: Flight tile directory.

        Raises:
            OSError / json.JSONDecodeError: On a missing or malformed file.
            ValueError: If a sidecar field or index row is invalid.
        """
        self._root = Path(root)
        payload = _read_object(self._root / "source.json")
        extra = sorted(set(payload) - set(_SOURCE_KEYS))
        if extra or any(key not in payload for key in _SOURCE_KEYS):
            raise ValueError(f"source.json keys must be {_SOURCE_KEYS}")
        self.band_names = _string_tuple(payload["band_names"], "band_names")
        bit_depth = payload["bit_depth"]
        if isinstance(bit_depth, bool) or not isinstance(bit_depth, int) or bit_depth < 1:
            raise ValueError("bit_depth must be an integer >= 1")
        self.bit_depth = bit_depth
        source_ref = payload["source_ref"]
        if not isinstance(source_ref, str):
            raise ValueError("source_ref must be a string")
        self.source_ref = source_ref
        reference = payload["gsd_reference_m"]
        if isinstance(reference, bool) or not isinstance(reference, int | float):
            raise ValueError("gsd_reference_m must be a number")
        if not math.isfinite(float(reference)) or float(reference) <= 0.0:
            raise ValueError("gsd_reference_m must be finite and > 0")
        self.gsd_reference_m = float(reference)
        self._refs = _load_index(self._root / "index.jsonl")

    def index(self) -> tuple[RawTileRef, ...]:
        """Return the index in file order.

        Returns:
            tuple[RawTileRef, ...]: One ref per ``index.jsonl`` row.
        """
        return self._refs

    def iter_tiles(self) -> Iterator[RawTile]:
        """Yield tiles in index order, reading one ``.npy`` at a time.

        Returns:
            Iterator[RawTile]: One forward pass.

        Raises:
            ValueError: If an image or a required mask is missing or has the
                wrong shape.
        """
        height, width = tile_hw()
        for ref in self._refs:
            image_path = self._root / "tiles" / f"{ref.tile_id}.npy"
            if not image_path.is_file():
                raise ValueError(f"missing tile array {image_path}")
            image = np.load(image_path)
            if image.dtype != np.uint16 or image.shape != (3, height, width):
                raise ValueError(f"{ref.tile_id} image must be uint16 (3, {height}, {width})")
            mask: np.ndarray | None = None
            if ref.has_mask:
                mask_path = self._root / "masks" / f"{ref.tile_id}.npy"
                if not mask_path.is_file():
                    raise ValueError(f"missing mask {mask_path}")
                stored = np.load(mask_path)
                if stored.dtype != np.uint8 or stored.shape != (height, width):
                    raise ValueError(f"{ref.tile_id} mask must be uint8 ({height}, {width})")
                mask = stored.reshape(1, height, width)
            yield RawTile(ref=ref, image=image, mask=mask)


def _validate_write(
    tile: FlightTileWrite,
    height: int,
    width: int,
    seen: set[str],
) -> None:
    """Raise ValueError when a tile cannot be written.

    Args:
        tile: Tile to store.
        height: Required H.
        width: Required W.
        seen: Tile ids already written.

    Returns:
        None.

    Raises:
        ValueError: On a bad id, grid index, GSD, image, or mask.
    """
    if (
        not tile.tile_id
        or tile.tile_id in seen
        or tile.tile_id in (".", "..")
        or "/" in tile.tile_id
        or "\\" in tile.tile_id
    ):
        raise ValueError(f"tile_id must be a unique file stem; got {tile.tile_id!r}")
    if not tile.frame_id:
        raise ValueError("frame_id must be non-empty")
    if not (0 <= tile.row < 8 and 0 <= tile.col < 8):
        raise ValueError(f"row and col must lie in 0..7; got {tile.row}, {tile.col}")
    if not math.isfinite(tile.label) or not math.isfinite(tile.theta_g_deg):
        raise ValueError("label and theta_g_deg must be finite")
    if not isinstance(tile.gsd_nominal, bool):
        raise ValueError("gsd_nominal must be a boolean")
    if not math.isfinite(tile.gsd.lateral_m) or tile.gsd.lateral_m <= 0.0:
        raise ValueError("gsd lateral_m must be finite and > 0")
    if not math.isfinite(tile.gsd.along_m) or tile.gsd.along_m <= 0.0:
        raise ValueError("gsd along_m must be finite and > 0")
    if tile.image.dtype != np.uint16 or tile.image.shape != (3, height, width):
        raise ValueError(f"image must be uint16 (3, {height}, {width})")
    if tile.mask is not None:
        _mask_hw(tile.mask, height, width)
    if tile.group_id is not None and not tile.group_id:
        raise ValueError("group_id must be non-empty when set")


def _mask_hw(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    """Return a uint8 ``(H, W)`` mask.

    Args:
        mask: np.ndarray[uint8, (H, W)] or ``(1, H, W)``.
        height: Required H.
        width: Required W.

    Returns:
        np.ndarray[uint8, (H, W)].

    Raises:
        ValueError: If the dtype or shape does not match.
    """
    if mask.dtype != np.uint8:
        raise ValueError("mask must be uint8")
    if mask.shape == (height, width):
        return mask
    if mask.shape == (1, height, width):
        return np.asarray(mask[0])
    raise ValueError(f"mask must be {(height, width)} or {(1, height, width)}; got {mask.shape}")


def _load_index(path: Path) -> tuple[RawTileRef, ...]:
    """Parse ``index.jsonl`` into raw refs.

    Args:
        path: JSONL path.

    Returns:
        tuple[RawTileRef, ...]: One ref per non-empty line.

    Raises:
        OSError / json.JSONDecodeError: On a missing or malformed file.
        ValueError: If a row is missing a field or has the wrong type.
    """
    refs: list[RawTileRef] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise ValueError("index.jsonl lines must be objects")
        allowed = set(_INDEX_REQUIRED) | set(_INDEX_OPTIONAL)
        extra = sorted(set(raw) - allowed)
        missing = [key for key in _INDEX_REQUIRED if key not in raw]
        if extra or missing:
            raise ValueError(f"index.jsonl keys invalid; missing {missing} extra {extra}")
        tile_id = _require_str(raw["tile_id"], "tile_id")
        if tile_id in (".", "..") or "/" in tile_id or "\\" in tile_id:
            raise ValueError("tile_id must be a file stem")
        if tile_id in seen:
            raise ValueError(f"duplicate tile_id {tile_id}")
        seen.add(tile_id)
        frame_id = _require_str(raw["frame_id"], "frame_id")
        group_raw = raw.get("group_id", frame_id)
        group_id = _require_str(group_raw, "group_id")
        row, col = _require_int(raw["row"], "row"), _require_int(raw["col"], "col")
        if row >= 8 or col >= 8:
            raise ValueError("flight grid indices must lie in 0..7")
        theta_g_deg = _require_float(raw["theta_g_deg"], "theta_g_deg")
        gsd_nominal = _require_bool(raw.get("gsd_nominal", False), "gsd_nominal")
        nearest = min(_ELEVATION_BINS, key=lambda elevation: abs(theta_g_deg - elevation))
        refs.append(
            RawTileRef(
                tile_id=tile_id,
                group_id=group_id,
                label=_require_float(raw["label"], "label"),
                has_mask=_require_bool(raw["has_mask"], "has_mask"),
                gsd=GsdPair(
                    lateral_m=_require_positive(raw["gsd_lateral_m"], "gsd_lateral_m"),
                    along_m=_require_positive(raw["gsd_along_m"], "gsd_along_m"),
                ),
                frame_id=frame_id,
                grid_rc=(row, col),
                bin_id=f"elevation{nearest}",
                theta_g_deg=theta_g_deg,
                gsd_nominal=gsd_nominal,
            )
        )
    if not refs:
        raise ValueError("index.jsonl is empty")
    return tuple(refs)


def _read_object(path: Path) -> dict[str, object]:
    """Parse a JSON object.

    Args:
        path: File path.

    Returns:
        dict[str, object]: Decoded mapping.

    Raises:
        OSError / json.JSONDecodeError: On a missing or malformed file.
        ValueError: If the root is not an object.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path.name} must be a JSON object")
    decoded: dict[str, object] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            raise ValueError(f"{path.name} keys must be strings")
        decoded[key] = value
    return decoded


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    """Return a non-empty tuple of strings.

    Args:
        value: Candidate list.
        name: Field name used in the error.

    Returns:
        tuple[str, ...]: Copied strings.

    Raises:
        ValueError: If the value is empty or not a list of strings.
    """
    if not isinstance(value, list) or len(value) < 1:
        raise ValueError(f"{name} must be a non-empty list of strings")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{name} must be a non-empty list of strings")
        items.append(item)
    return tuple(items)


def _require_str(value: object, name: str) -> str:
    """Return a non-empty string.

    Args:
        value: Candidate.
        name: Field name.

    Returns:
        str: The string.

    Raises:
        ValueError: If the value is not a non-empty string.
    """
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_bool(value: object, name: str) -> bool:
    """Return a JSON boolean.

    Args:
        value: Candidate.
        name: Field name.

    Returns:
        bool: The boolean.

    Raises:
        ValueError: If the value is not a boolean.
    """
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _require_int(value: object, name: str) -> int:
    """Return a non-negative integer.

    Args:
        value: Candidate.
        name: Field name.

    Returns:
        int: The integer.

    Raises:
        ValueError: If the value is a boolean or is not an integer >= 0.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be an integer >= 0")
    return value


def _require_float(value: object, name: str) -> float:
    """Return a finite float.

    Args:
        value: Candidate number.
        name: Field name.

    Returns:
        float: The value.

    Raises:
        ValueError: If the value is not a finite number.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _require_positive(value: object, name: str) -> float:
    """Return a finite float greater than 0.

    Args:
        value: Candidate number.
        name: Field name.

    Returns:
        float: The value.

    Raises:
        ValueError: If the value is not finite and greater than 0.
    """
    number = _require_float(value, name)
    if number <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return number
