"""Shared fixtures for tools.ml_models integration tests.

The ``build_synthetic_dataset`` fixture returns a builder that writes a
small planted-blob finished dataset at the flight tile size. Tiles are
float32 unit pixels; a positive tile carries a bright block and every tile
carries a mask, so any split assignment yields segmentor rows in every split.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

import numpy as np
import pytest
from flight.libs.config import InferenceConfig
from tools.ml_models.dataset.build import build_dataset
from tools.ml_models.dataset.raw import BinSpec, GsdPair, RawTile, RawTileRef
from tools.ml_models.dataset.spec import BuildSpec

_DEFAULT = InferenceConfig()
_TILE_HW = (
    _DEFAULT.input_height_px // _DEFAULT.tile_rows,
    _DEFAULT.input_width_px // _DEFAULT.tile_cols,
)

_WINDOW: tuple[GsdPair, ...] = (
    GsdPair(_DEFAULT.gsd_reference_m, _DEFAULT.gsd_reference_m),
    GsdPair(16.5, 17.1),
    GsdPair(18.6, 22.0),
    GsdPair(23.3, 35.8),
)


class _DatasetBuilder(Protocol):
    def __call__(self, dest: Path, *, n: int = 9, seed: int = 0) -> Path: ...


class _BlobSource:
    """In-memory planted-blob ``RawSource`` for finished-dataset fixtures."""

    name = "fixture"
    band_names = _DEFAULT.input_bands
    domain = "unit"
    source_ref = ""
    bins: tuple[BinSpec, ...] = ()

    def __init__(self, n: int, seed: int) -> None:
        """Generate ``n`` tiles. ``n`` must be at least 3 for a group split."""
        if n < 3:
            raise ValueError(f"fixture source needs at least 3 tiles; got {n}")
        self._tiles = _generate(n, seed)

    def index(self) -> tuple[RawTileRef, ...]:
        """Return the generated refs in stream order."""
        return tuple(tile.ref for tile in self._tiles)

    def iter_tiles(self) -> Iterator[RawTile]:
        """Yield the generated tiles in index order."""
        yield from self._tiles


def _generate(n: int, seed: int) -> tuple[RawTile, ...]:
    """Build ``n`` planted-blob tiles at the flight size."""
    height, width = _TILE_HW
    n_groups = 4 if n >= 4 else n
    generator = np.random.default_rng(seed)
    tiles: list[RawTile] = []
    for index in range(n):
        chosen = 1.0 if index % 2 == 0 else 0.0
        gsd = _WINDOW[index % len(_WINDOW)]
        image = generator.random((3, height, width), dtype=np.float32) * np.float32(0.2)
        blob = (slice(height // 4, height // 2), slice(width // 4, width // 2))
        if chosen >= 0.5:
            image[:, blob[0], blob[1]] = np.float32(0.9)
        mask = np.zeros((1, height, width), dtype=np.uint8)
        if chosen >= 0.5:
            mask[0, blob[0], blob[1]] = np.uint8(1)
        group = f"g{index % n_groups}"
        tiles.append(
            RawTile(
                ref=RawTileRef(
                    tile_id=f"fx-{index:04d}",
                    group_id=group,
                    label=chosen,
                    has_mask=True,
                    gsd=gsd,
                    height=height,
                    width=width,
                    frame_id=group,
                    grid_rc=(index // _DEFAULT.tile_cols, index % _DEFAULT.tile_cols),
                    bin_id="",
                ),
                image=image,
                mask=mask,
            )
        )
    return tuple(tiles)


@pytest.fixture
def build_synthetic_dataset() -> _DatasetBuilder:
    """Return a builder that writes a small planted-blob finished dataset."""

    def _build(dest: Path, *, n: int = 9, seed: int = 0) -> Path:
        build_dataset(_BlobSource(n, seed), dest, BuildSpec())
        return dest

    return _build
