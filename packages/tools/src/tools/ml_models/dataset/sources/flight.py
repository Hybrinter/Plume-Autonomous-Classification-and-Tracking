"""Ground-side reader and fixture writer for the flight tile import format.

Contains:
  - FlightTileWrite: one tile accepted by the writer.
  - write_flight_tile_dir: create ``source.json``, ``index.jsonl``, ``tiles/``,
    and ``masks/``.
  - FlightTileDir: ``RawSource`` reader.

This module describes imported finished tiles; it does not implement onboard
frame collection or storage policy. On-disk images are float32 unit pixels
``(C, H, W)`` in ``band_names`` order. Masks are uint8 ``(H, W)``. A missing
``group_id`` in ``index.jsonl`` defaults to ``frame_id``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import cast

import numpy as np
from flight.libs.config import InferenceConfig
from flight.libs.types import Err
from flight.payload.preprocess.tile_product import (
    UNIT_TILE_SOURCE_SCHEMA,
    UnitTileCapture,
    UnitTileLayout,
    validate_capture,
    validate_layout,
    validate_unit_tile,
)
from pydantic import TypeAdapter

from tools.ml_models.dataset.raw import (
    BinSpec,
    GsdPair,
    ObservationMetadata,
    RawTile,
    RawTileRef,
)

_OBSERVATION_METADATA = TypeAdapter(ObservationMetadata)

SCHEMA_VERSION = UNIT_TILE_SOURCE_SCHEMA
_DEFAULT = InferenceConfig()
_DEFAULT_TILE_HW = (
    _DEFAULT.input_height_px // _DEFAULT.tile_rows,
    _DEFAULT.input_width_px // _DEFAULT.tile_cols,
)
_DEFAULT_GRID = (_DEFAULT.tile_rows, _DEFAULT.tile_cols)

_SOURCE_KEYS: tuple[str, ...] = (
    "schema",
    "domain",
    "image_dtype",
    "band_names",
    "tile_hw",
    "grid",
    "source_ref",
    "gsd_reference_m",
)
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
_INDEX_OPTIONAL: tuple[str, ...] = ("group_id", "gsd_nominal", "metadata")
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
        image: np.ndarray[float32, (len(band_names), H, W)] unit pixels.
        mask: np.ndarray[uint8, (H, W)] or ``(1, H, W)``. None when
            the tile has no mask.
        group_id: Split group. None stores ``frame_id``.
        gsd_nominal: True when ``gsd`` is nominal orbit geometry rather than
            measured capture geometry.
        metadata: Authoritative observation provenance. ``observation_id``
            must be the original ``tile_id`` or None; the reader fills None
            with ``tile_id``. Dates stay None unless explicitly recorded.
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
    metadata: ObservationMetadata = field(default_factory=ObservationMetadata)


def write_flight_tile_dir(
    dest: str | Path,
    tiles: tuple[FlightTileWrite, ...] | list[FlightTileWrite],
    *,
    band_names: tuple[str, ...] = _DEFAULT.input_bands,
    tile_hw: tuple[int, int] = _DEFAULT_TILE_HW,
    grid: tuple[int, int] = _DEFAULT_GRID,
    source_ref: str = "",
    gsd_reference_m: float = _DEFAULT.gsd_reference_m,
) -> None:
    """Write a flight tile directory.

    Args:
        dest: New directory.
        tiles: Tiles in index order.
        band_names: Channel names stored in ``source.json``.
        tile_hw: ``(height, width)`` stored in ``source.json``.
        grid: ``(rows, cols)`` stored in ``source.json``.
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
    layout = UnitTileLayout(
        band_names=band_names,
        tile_hw=tile_hw,
        grid=grid,
        gsd_reference_m=gsd_reference_m,
    )
    if isinstance(validate_layout(layout), Err):
        raise ValueError(f"invalid flight tile layout {layout!r}")
    height, width = layout.tile_hw
    grid_rows, grid_cols = layout.grid
    if len(tiles) < 1:
        raise ValueError("flight tile directory requires at least one tile")
    root.mkdir()
    (root / "tiles").mkdir()
    (root / "masks").mkdir()
    source_payload = {
        "schema": SCHEMA_VERSION,
        "domain": "unit",
        "image_dtype": "float32",
        "band_names": list(band_names),
        "tile_hw": [height, width],
        "grid": [grid_rows, grid_cols],
        "source_ref": source_ref,
        "gsd_reference_m": gsd_reference_m,
    }
    (root / "source.json").write_text(
        json.dumps(source_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    lines: list[str] = []
    seen: set[str] = set()
    for tile in tiles:
        _validate_write(tile, layout, seen)
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
                    "metadata": _metadata_payload(tile.metadata),
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
        domain: Always ``unit``.
        source_ref: From ``source.json``.
        bins: Always empty.
        tile_hw: Recorded ``(height, width)``.
        grid: Recorded ``(rows, cols)``.
        gsd_reference_m: From ``source.json``.
        layout: Validated ``UnitTileLayout`` for the recorded layout.
    """

    name = "flight"
    domain = "unit"
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
        if payload.get("schema") != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported flight tile source schema {payload.get('schema')!r}; "
                f"rewrite the tile directory with schema {SCHEMA_VERSION}"
            )
        extra = sorted(set(payload) - set(_SOURCE_KEYS))
        if extra or any(key not in payload for key in _SOURCE_KEYS):
            raise ValueError(f"source.json keys must be {_SOURCE_KEYS}")
        if payload["domain"] != "unit":
            raise ValueError("domain must be 'unit'")
        if payload["image_dtype"] != "float32":
            raise ValueError("image_dtype must be 'float32'")
        self.band_names = _string_tuple(payload["band_names"], "band_names")
        self.tile_hw = _positive_pair(payload["tile_hw"], "tile_hw")
        self.grid = _positive_pair(payload["grid"], "grid")
        source_ref = payload["source_ref"]
        if not isinstance(source_ref, str):
            raise ValueError("source_ref must be a string")
        self.source_ref = source_ref
        reference = payload["gsd_reference_m"]
        if isinstance(reference, bool) or not isinstance(reference, int | float):
            raise ValueError("gsd_reference_m must be a number")
        self.layout = UnitTileLayout(
            band_names=self.band_names,
            tile_hw=self.tile_hw,
            grid=self.grid,
            gsd_reference_m=reference,
        )
        if isinstance(validate_layout(self.layout), Err):
            raise ValueError(f"source.json layout is invalid: {self.layout!r}")
        self.gsd_reference_m = float(self.layout.gsd_reference_m)
        self._refs = _load_index(self._root / "index.jsonl", self.layout)

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
                wrong dtype or shape.
        """
        height, width = self.tile_hw
        for ref in self._refs:
            image_path = self._root / "tiles" / f"{ref.tile_id}.npy"
            if not image_path.is_file():
                raise ValueError(f"missing tile array {image_path}")
            image = np.load(image_path)
            expected_shape = (len(self.band_names), height, width)
            if isinstance(validate_unit_tile(image, self.layout), Err):
                raise ValueError(
                    f"{ref.tile_id} image must be float32 {expected_shape} finite inside [0, 1]"
                )
            mask: np.ndarray | None = None
            if ref.has_mask:
                mask_path = self._root / "masks" / f"{ref.tile_id}.npy"
                if not mask_path.is_file():
                    raise ValueError(f"missing mask {mask_path}")
                stored = np.load(mask_path)
                if (
                    stored.dtype != np.uint8
                    or stored.shape != (height, width)
                    or not np.all((stored == 0) | (stored == 1))
                ):
                    raise ValueError(f"{ref.tile_id} mask must be binary uint8 ({height}, {width})")
                mask = stored.reshape(1, height, width)
            yield RawTile(ref=ref, image=image, mask=mask)


def _validate_write(tile: FlightTileWrite, layout: UnitTileLayout, seen: set[str]) -> None:
    """Raise ValueError when a tile cannot be written.

    Args:
        tile: Tile to store.
        layout: Validated unit-tile layout.
        seen: Tile ids already written.

    Returns:
        None.

    Raises:
        ValueError: On a duplicate id, invalid capture, image, label, or mask.
    """
    if tile.tile_id in seen:
        raise ValueError(f"tile_id must be a unique file stem; got {tile.tile_id!r}")
    capture = UnitTileCapture(
        tile_id=tile.tile_id,
        frame_id=tile.frame_id,
        grid_rc=(tile.row, tile.col),
        theta_g_deg=tile.theta_g_deg,
        gsd=tile.gsd,
        gsd_nominal=tile.gsd_nominal,
    )
    if isinstance(validate_capture(capture, layout), Err):
        raise ValueError(f"invalid flight tile capture {capture!r}")
    if isinstance(tile.label, bool) or tile.label not in (0.0, 1.0):
        raise ValueError(f"label must be finite and exactly 0 or 1; got {tile.label!r}")
    height, width = layout.tile_hw
    expected_shape = (len(layout.band_names), height, width)
    if isinstance(validate_unit_tile(tile.image, layout), Err):
        raise ValueError(f"image must be float32 {expected_shape} finite inside [0, 1]")
    if tile.mask is not None:
        mask = _mask_hw(tile.mask, height, width)
        if not np.all((mask == 0) | (mask == 1)):
            raise ValueError("mask must contain binary pixels")
    if tile.group_id is not None and not tile.group_id:
        raise ValueError("group_id must be non-empty when set")
    if not isinstance(tile.metadata, ObservationMetadata):
        raise ValueError("metadata must be an ObservationMetadata")
    if tile.metadata.observation_id not in (None, tile.tile_id):
        raise ValueError(
            f"metadata observation_id must equal the original tile_id {tile.tile_id!r}"
        )


def _metadata_payload(metadata: ObservationMetadata) -> dict[str, object]:
    """Serialize validated observation metadata for ``index.jsonl``.

    Args:
        metadata: Metadata attached to the tile.

    Returns:
        dict[str, object]: JSON object form.

    Raises:
        ValueError: If the value is not a valid ObservationMetadata.
    """
    try:
        validated = _OBSERVATION_METADATA.validate_python(asdict(metadata))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"metadata is invalid: {exc}") from exc
    return cast(
        dict[str, object],
        _OBSERVATION_METADATA.dump_python(validated, mode="json"),
    )


def _index_metadata(value: object, tile_id: str) -> ObservationMetadata:
    """Parse one present ``metadata`` index field.

    Args:
        value: JSON value under the ``metadata`` key.
        tile_id: Original tile id used when no observation_id is recorded.

    Returns:
        ObservationMetadata: Parsed metadata with a non-null observation_id.

    Raises:
        ValueError: If the value is malformed or names a different tile.
    """
    try:
        metadata = _OBSERVATION_METADATA.validate_python(value)
    except ValueError as exc:
        raise ValueError(f"index.jsonl metadata is invalid: {exc}") from exc
    if metadata.observation_id is None:
        return replace(metadata, observation_id=tile_id)
    if metadata.observation_id != tile_id:
        raise ValueError(f"index.jsonl metadata observation_id must equal tile_id {tile_id!r}")
    return metadata


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


def _load_index(path: Path, layout: UnitTileLayout) -> tuple[RawTileRef, ...]:
    """Parse ``index.jsonl`` into raw refs.

    Args:
        path: JSONL path.
        layout: Validated unit-tile layout the captures are checked against.

    Returns:
        tuple[RawTileRef, ...]: One ref per non-empty line.

    Raises:
        OSError / json.JSONDecodeError: On a missing or malformed file.
        ValueError: If a row is missing a field, has the wrong type, or fails
            the shared capture contract.
    """
    height, width = layout.tile_hw
    refs: list[RawTileRef] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
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
        if tile_id in seen:
            raise ValueError(f"duplicate tile_id {tile_id}")
        seen.add(tile_id)
        frame_id = _require_str(raw["frame_id"], "frame_id")
        group_raw = raw.get("group_id", frame_id)
        group_id = _require_str(group_raw, "group_id")
        row, col = _require_int(raw["row"], "row"), _require_int(raw["col"], "col")
        theta_g_deg = _require_float(raw["theta_g_deg"], "theta_g_deg")
        gsd_nominal = _require_bool(raw.get("gsd_nominal", False), "gsd_nominal")
        label = _require_float(raw["label"], "label")
        if label not in (0.0, 1.0):
            raise ValueError(f"index.jsonl label must be exactly 0 or 1; got {label!r}")
        has_mask = _require_bool(raw["has_mask"], "has_mask")
        capture = UnitTileCapture(
            tile_id=tile_id,
            frame_id=frame_id,
            grid_rc=(row, col),
            theta_g_deg=theta_g_deg,
            gsd=GsdPair(
                lateral_m=_require_float(raw["gsd_lateral_m"], "gsd_lateral_m"),
                along_m=_require_float(raw["gsd_along_m"], "gsd_along_m"),
            ),
            gsd_nominal=gsd_nominal,
        )
        if isinstance(validate_capture(capture, layout), Err):
            raise ValueError(
                f"index.jsonl line {line_number} is not a valid unit tile capture: {raw!r}"
            )
        nearest = min(_ELEVATION_BINS, key=lambda elevation: abs(theta_g_deg - elevation))
        refs.append(
            RawTileRef(
                tile_id=tile_id,
                group_id=group_id,
                label=label,
                has_mask=has_mask,
                gsd=capture.gsd,
                height=height,
                width=width,
                frame_id=frame_id,
                grid_rc=(row, col),
                bin_id=f"elevation{nearest}",
                theta_g_deg=theta_g_deg,
                gsd_nominal=gsd_nominal,
                metadata=(
                    ObservationMetadata(observation_id=tile_id)
                    if "metadata" not in raw
                    else _index_metadata(raw["metadata"], tile_id)
                ),
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


def _positive_pair(value: object, name: str) -> tuple[int, int]:
    """Return a pair of positive integers.

    Args:
        value: Candidate two-element list.
        name: Field name used in the error.

    Returns:
        tuple[int, int]: Parsed pair.

    Raises:
        ValueError: If the value is not a pair of integers >= 1.
    """
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ValueError(f"{name} must be a pair of integers >= 1")
    first, second = value
    for item in (first, second):
        if isinstance(item, bool) or not isinstance(item, int) or item < 1:
            raise ValueError(f"{name} must be a pair of integers >= 1")
    return (first, second)


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
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number
