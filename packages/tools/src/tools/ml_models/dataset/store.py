"""Shard writer and reader for a finished dataset.

Contains:
  - RowRecord: one ``rows.jsonl`` object.
  - ShardWriter: preallocated ``.npy`` memmaps plus a row list.
  - read_rows, read_images, read_gsd, read_labels, read_masks.

Images are float32 unit pixels ``(N, C, H, W)``. GSD is float32 metres
``(N, 2)``. Labels are float32 ``(N, 1)``. Segmentor shards also store uint8
masks ``(N, 1, H, W)``.
The writer fills a temporary shard directory. The caller renames the dataset
root after every shard has been closed.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_ROW_KEYS: tuple[str, ...] = (
    "tile_id",
    "group_id",
    "frame_id",
    "grid_rc",
    "bin_id",
    "element",
    "theta_g_deg",
    "gsd_nominal",
)

_ROW_REQUIRED: tuple[str, ...] = _ROW_KEYS[:6]


@dataclass(frozen=True, slots=True)
class RowRecord:
    """One stored row.

    Attributes:
        tile_id: Source tile id.
        group_id: Split group.
        frame_id: Flight frame id, or None.
        grid_rc: ``(row, col)`` or None.
        bin_id: GSD bin name. Empty when the source has no bins.
        element: Dihedral element. Val and test store ``id``.
        theta_g_deg: Gimbal elevation in degrees, or None when unrecorded.
        gsd_nominal: True when the row GSD is nominal orbit geometry.
    """

    tile_id: str
    group_id: str
    frame_id: str | None
    grid_rc: tuple[int, int] | None
    bin_id: str
    element: str
    theta_g_deg: float | None = None
    gsd_nominal: bool = False


class ShardWriter:
    """Fill one preallocated shard directory.

    Attributes:
        directory: Shard path ``<task>/<split>/<H>x<W>``.
    """

    def __init__(
        self,
        directory: Path,
        count: int,
        height: int,
        width: int,
        *,
        channels: int,
        with_masks: bool,
    ) -> None:
        """Allocate memmaps for ``count`` rows.

        Args:
            directory: Empty destination directory. Created here.
            count: Row count. Must be at least 1.
            height: Tile H.
            width: Tile W.
            channels: Image channel count. Must be at least 1.
            with_masks: When True, allocate ``masks.npy``.

        Raises:
            ValueError: If ``count``, ``height``, ``width``, or ``channels``
                is below 1.
        """
        if count < 1 or height < 1 or width < 1 or channels < 1:
            raise ValueError(
                f"shard shape must be positive; got n={count} c={channels} {height}x{width}"
            )
        directory.mkdir(parents=True, exist_ok=False)
        self.directory = directory
        self._count = count
        self._channels = channels
        self._height = height
        self._width = width
        self._cursor = 0
        self._rows: list[RowRecord] = []
        self._images = np.lib.format.open_memmap(
            directory / "images.npy",
            mode="w+",
            dtype=np.float32,
            shape=(count, channels, height, width),
        )
        self._gsd = np.lib.format.open_memmap(
            directory / "gsd.npy",
            mode="w+",
            dtype=np.float32,
            shape=(count, 2),
        )
        self._labels = np.lib.format.open_memmap(
            directory / "labels.npy",
            mode="w+",
            dtype=np.float32,
            shape=(count, 1),
        )
        self._masks: np.memmap | None
        if with_masks:
            self._masks = np.lib.format.open_memmap(
                directory / "masks.npy",
                mode="w+",
                dtype=np.uint8,
                shape=(count, 1, height, width),
            )
        else:
            self._masks = None

    def append(
        self,
        image: np.ndarray,
        gsd_m: np.ndarray,
        label: float,
        mask: np.ndarray | None,
        row: RowRecord,
    ) -> None:
        """Write the next row.

        Args:
            image: np.ndarray[float32, (C, H, W)] unit pixels.
            gsd_m: np.ndarray[float32, (2,)] lateral then along-track metres.
            label: Classification target.
            mask: np.ndarray[uint8, (1, H, W)] for a segmentor shard, else None.
            row: Identity written to ``rows.jsonl``.

        Returns:
            None.

        Raises:
            ValueError: If the shard is full or an array has the wrong shape.
            RuntimeError: If this is a segmentor shard and ``mask`` is None, or
                a classifier shard and ``mask`` is present.
        """
        if self._cursor >= self._count:
            raise ValueError(f"shard {self.directory} is full")
        if image.shape != (self._channels, self._height, self._width) or image.dtype != np.float32:
            raise ValueError(
                f"image must be float32 ({self._channels}, {self._height}, {self._width}); "
                f"got {image.dtype} {image.shape}"
            )
        gsd = np.asarray(gsd_m, dtype=np.float32)
        if gsd.shape != (2,):
            raise ValueError(f"gsd must have shape (2,); got {gsd.shape}")
        index = self._cursor
        self._images[index] = image
        self._gsd[index] = gsd
        self._labels[index, 0] = np.float32(label)
        if self._masks is None:
            if mask is not None:
                raise RuntimeError("classifier shard received a mask")
        else:
            if mask is None:
                raise RuntimeError("segmentor shard is missing a mask")
            if mask.shape != (1, self._height, self._width) or mask.dtype != np.uint8:
                raise ValueError(
                    f"mask must be uint8 (1, {self._height}, {self._width}); "
                    f"got {mask.dtype} {mask.shape}"
                )
            self._masks[index] = mask
        self._rows.append(row)
        self._cursor += 1

    def close(self) -> None:
        """Flush arrays and write ``rows.jsonl``.

        Returns:
            None.

        Raises:
            RuntimeError: If fewer rows were appended than allocated.
        """
        if self._cursor != self._count:
            raise RuntimeError(f"shard {self.directory} wrote {self._cursor} of {self._count} rows")
        self._images.flush()
        self._gsd.flush()
        self._labels.flush()
        del self._images
        del self._gsd
        del self._labels
        if self._masks is not None:
            self._masks.flush()
            del self._masks
            self._masks = None
        lines = [json.dumps(_row_payload(row), separators=(",", ":")) for row in self._rows]
        (self.directory / "rows.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def abort(self) -> None:
        """Drop memmap handles without writing ``rows.jsonl``.

        Returns:
            None.

        Notes:
            Called when a build fails so the temporary directory can be removed
            while no arrays are still mapped.
        """
        for name in ("_images", "_gsd", "_labels", "_masks"):
            if getattr(self, name, None) is not None:
                delattr(self, name)


def read_rows(shard_dir: Path) -> tuple[RowRecord, ...]:
    """Load ``rows.jsonl``.

    Args:
        shard_dir: Shard directory.

    Returns:
        tuple[RowRecord, ...]: Rows in file order.

    Raises:
        OSError / json.JSONDecodeError: On a missing or malformed file.
        ValueError: If a line is not an object with the expected keys.
    """
    path = shard_dir / "rows.jsonl"
    rows: list[RowRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise ValueError("rows.jsonl lines must be objects")
        extra = sorted(set(raw) - set(_ROW_KEYS))
        if extra or any(key not in raw for key in _ROW_REQUIRED):
            raise ValueError(f"rows.jsonl keys must be {_ROW_KEYS}")
        rows.append(_row_from_payload(raw))
    return tuple(rows)


def read_images(shard_dir: Path) -> np.ndarray:
    """Load ``images.npy``.

    Args:
        shard_dir: Shard directory.

    Returns:
        np.ndarray[float32, (N, C, H, W)].
    """
    return np.asarray(np.load(shard_dir / "images.npy"))


def read_gsd(shard_dir: Path) -> np.ndarray:
    """Load ``gsd.npy``.

    Args:
        shard_dir: Shard directory.

    Returns:
        np.ndarray[float32, (N, 2)].
    """
    return np.asarray(np.load(shard_dir / "gsd.npy"))


def read_labels(shard_dir: Path) -> np.ndarray:
    """Load ``labels.npy``.

    Args:
        shard_dir: Shard directory.

    Returns:
        np.ndarray[float32, (N, 1)].
    """
    return np.asarray(np.load(shard_dir / "labels.npy"))


def read_masks(shard_dir: Path) -> np.ndarray | None:
    """Load ``masks.npy`` when the shard has one.

    Args:
        shard_dir: Shard directory.

    Returns:
        np.ndarray[uint8, (N, 1, H, W)] or None when the file is absent.
    """
    path = shard_dir / "masks.npy"
    if not path.is_file():
        return None
    return np.asarray(np.load(path))


def _row_payload(row: RowRecord) -> dict[str, object]:
    """Convert a row to a JSON object.

    Args:
        row: Stored identity.

    Returns:
        dict[str, object]: Keys in ``_ROW_KEYS`` order.
    """
    grid: list[int] | None = None if row.grid_rc is None else [row.grid_rc[0], row.grid_rc[1]]
    return {
        "tile_id": row.tile_id,
        "group_id": row.group_id,
        "frame_id": row.frame_id,
        "grid_rc": grid,
        "bin_id": row.bin_id,
        "element": row.element,
        "theta_g_deg": row.theta_g_deg,
        "gsd_nominal": row.gsd_nominal,
    }


def _row_from_payload(raw: dict[str, object]) -> RowRecord:
    """Parse one ``rows.jsonl`` object.

    Args:
        raw: Decoded JSON object with the expected keys.

    Returns:
        RowRecord: Parsed row.

    Raises:
        ValueError: If a field has the wrong type.
    """
    tile_id = raw["tile_id"]
    group_id = raw["group_id"]
    frame_id = raw["frame_id"]
    grid = raw["grid_rc"]
    bin_id = raw["bin_id"]
    element = raw["element"]
    theta_g_deg = raw.get("theta_g_deg")
    gsd_nominal = raw.get("gsd_nominal", False)
    if not isinstance(tile_id, str) or not isinstance(group_id, str):
        raise ValueError("tile_id and group_id must be strings")
    if frame_id is not None and not isinstance(frame_id, str):
        raise ValueError("frame_id must be a string or null")
    if not isinstance(bin_id, str) or not isinstance(element, str):
        raise ValueError("bin_id and element must be strings")
    if theta_g_deg is not None:
        if isinstance(theta_g_deg, bool) or not isinstance(theta_g_deg, int | float):
            raise ValueError("theta_g_deg must be a number or null")
        theta_g_deg = float(theta_g_deg)
        if not math.isfinite(theta_g_deg):
            raise ValueError("theta_g_deg must be finite")
    if not isinstance(gsd_nominal, bool):
        raise ValueError("gsd_nominal must be a boolean")
    return RowRecord(
        tile_id=tile_id,
        group_id=group_id,
        frame_id=frame_id,
        grid_rc=_grid_rc(grid),
        bin_id=bin_id,
        element=element,
        theta_g_deg=theta_g_deg,
        gsd_nominal=gsd_nominal,
    )


def _grid_rc(value: object) -> tuple[int, int] | None:
    """Parse a grid coordinate.

    Args:
        value: JSON null or a two-integer list.

    Returns:
        tuple[int, int] | None: Row and column, or None.

    Raises:
        ValueError: If the value is not null or a pair of integers.
    """
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("grid_rc must be null or [row, col]")
    row, col = value
    if isinstance(row, bool) or isinstance(col, bool):
        raise ValueError("grid_rc must be null or [row, col]")
    if not isinstance(row, int) or not isinstance(col, int):
        raise ValueError("grid_rc must be null or [row, col]")
    return (row, col)
