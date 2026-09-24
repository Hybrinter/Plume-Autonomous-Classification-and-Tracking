"""Memmap of native GeoTIFF stacks from one archive pass.

Contains:
  - TileCache: stem-addressed float32 stacks.
  - build_cache: write the memmap from ``iter_stacks``.
  - open_cache: reopen a cache written by ``build_cache``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from tools.original_dataset_analysis.grid import NATIVE_SIDE
from tools.original_dataset_analysis.index import TileRef, iter_stacks

_META = "meta.json"
_STACKS = "stacks.dat"


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
        ValueError: If ``tiles`` is empty or a stack is not ``(C, 120, 120)``.
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
        array = np.asarray(stack, dtype=np.float32)
        if array.ndim != 3 or array.shape[1] != NATIVE_SIDE or array.shape[2] != NATIVE_SIDE:
            raise ValueError(f"expected (C, 120, 120); got {array.shape}")
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
