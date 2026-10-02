"""In-memory planted-blob tiles at the flight tile size.

Contains:
  - SyntheticSource: ``RawSource`` whose GSD values sample the science window.

Images are uint16 DN. A positive tile carries a bright block. Every third tile
carries a mask, including annotated negatives. The source does not touch disk.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from tools.ml_models.dataset.geometry import GRID_COLS, GSD_REFERENCE_M, INPUT_BANDS, tile_hw
from tools.ml_models.dataset.raw import BinSpec, GsdPair, RawTile, RawTileRef

_WINDOW: tuple[GsdPair, ...] = (
    GsdPair(GSD_REFERENCE_M, GSD_REFERENCE_M),
    GsdPair(16.5, 17.1),
    GsdPair(18.6, 22.0),
    GsdPair(23.3, 35.8),
)


class SyntheticSource:
    """Planted-blob tiles with GSD sampled across the elevation window.

    Attributes:
        name: Always ``synthetic``.
        band_names: Flight ``BLUE``, ``GREEN``, ``RED``.
        domain: Always ``dn``.
        bit_depth: Always 12.
        source_ref: Always empty.
        extent_m: Always None.
        bins: Always empty.
    """

    name = "synthetic"
    band_names = INPUT_BANDS
    domain = "dn"
    bit_depth = 12
    source_ref = ""
    extent_m: tuple[float, float] | None = None
    bins: tuple[BinSpec, ...] = ()

    def __init__(self, n: int = 12, seed: int = 0, *, label: float | None = None) -> None:
        """Generate ``n`` tiles.

        Args:
            n: Tile count. At least 3 so a group split is possible.
            seed: NumPy seed for the background noise.
            label: When set, every tile uses this label. Otherwise even indices
                are positive.

        Raises:
            ValueError: If ``n`` is below 3 or ``label`` is not finite.
        """
        if n < 3:
            raise ValueError(f"synthetic source needs at least 3 tiles; got {n}")
        if label is not None and not np.isfinite(label):
            raise ValueError("label must be finite")
        self._tiles = _generate(n, seed, label)

    def index(self) -> tuple[RawTileRef, ...]:
        """Return the generated refs in stream order.

        Returns:
            tuple[RawTileRef, ...]: One ref per tile.
        """
        return tuple(tile.ref for tile in self._tiles)

    def iter_tiles(self) -> Iterator[RawTile]:
        """Yield the generated tiles in index order.

        Returns:
            Iterator[RawTile]: One forward pass over the in-memory tiles.
        """
        yield from self._tiles


def _generate(n: int, seed: int, label: float | None) -> tuple[RawTile, ...]:
    """Build planted-blob tiles.

    Args:
        n: Tile count.
        seed: Noise seed.
        label: Optional constant label.

    Returns:
        tuple[RawTile, ...]: Tiles in index order.
    """
    height, width = tile_hw()
    n_groups = 4 if n >= 4 else n
    generator = np.random.default_rng(seed)
    tiles: list[RawTile] = []
    for index in range(n):
        chosen = float(label) if label is not None else (1.0 if index % 2 == 0 else 0.0)
        gsd = _WINDOW[index % len(_WINDOW)]
        image = generator.integers(0, 400, size=(3, height, width), dtype=np.uint16)
        blob = (
            slice(height // 4, height // 2),
            slice(width // 4, width // 2),
        )
        if chosen >= 0.5:
            image[:, blob[0], blob[1]] = np.uint16(3000)
        mask: np.ndarray | None = None
        if index % 3 == 0:
            stored = np.zeros((1, height, width), dtype=np.uint8)
            if chosen >= 0.5:
                stored[0, blob[0], blob[1]] = np.uint8(1)
            mask = stored
        group = f"g{index % n_groups}"
        tiles.append(
            RawTile(
                ref=RawTileRef(
                    tile_id=f"syn-{index:04d}",
                    group_id=group,
                    label=chosen,
                    has_mask=mask is not None,
                    gsd=gsd,
                    frame_id=group,
                    grid_rc=(index // GRID_COLS, index % GRID_COLS),
                    bin_id="",
                ),
                image=image,
                mask=mask,
            )
        )
    return tuple(tiles)
