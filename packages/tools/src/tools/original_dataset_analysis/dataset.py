"""Tile samples for the band study.

Contains:
  - TileReader: reads one native stack.
  - StudyDataset: selected bands, a legal side, and frozen moments.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from tools.original_dataset_analysis.bands import BandSubset
from tools.original_dataset_analysis.grid import coarsen, rasterize_mask
from tools.original_dataset_analysis.index import TileRef
from tools.original_dataset_analysis.normalize import BandStats, apply_band_stats

TileReader = Callable[[TileRef], np.ndarray]


class StudyDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """One tile at a fixed band subset and ground-sample side.

    Each item is ``(image, label, mask)``. ``label`` is a scalar float.
    ``mask`` has shape ``(1, side, side)``.
    """

    def __init__(
        self,
        tiles: Sequence[TileRef],
        reader: TileReader,
        subset: BandSubset,
        stats: BandStats,
        side_px: int,
        *,
        mask_rule: str = "half",
    ) -> None:
        """Store the tiles and the transforms applied on each read.

        Args:
            tiles: Tiles in this split.
            reader: Returns a native ``(bands, 120, 120)`` stack in file order.
            subset: Channels kept, in file order.
            stats: Frozen per-channel moments for ``subset``.
            side_px: Legal output side.
            mask_rule: ``touch`` or ``half``.

        Raises:
            ValueError: If stats do not match the subset width.
        """
        if stats.mean.shape[0] != len(subset.indices):
            raise ValueError(
                f"stats have {stats.mean.shape[0]} channels; subset has {len(subset.indices)}"
            )
        self._tiles = tuple(tiles)
        self._reader = reader
        self._subset = subset
        self._stats = stats
        self._side_px = side_px
        self._mask_rule = mask_rule

    def __len__(self) -> int:
        """Return the number of tiles."""
        return len(self._tiles)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return one normalized image, its presence label, and its mask.

        Args:
            index: Tile position.

        Returns:
            tuple: Image ``(C, side, side)``, label ``(1,)``, mask ``(1, side, side)``.
        """
        tile = self._tiles[index]
        native = np.asarray(self._reader(tile), dtype=np.float32)
        selected = native[list(self._subset.indices)]
        image = apply_band_stats(coarsen(selected, self._side_px), self._stats)
        polygons = () if tile.polygons is None else tile.polygons
        mask = rasterize_mask(polygons, self._side_px, rule=self._mask_rule)
        label = np.array([1.0 if tile.positive else 0.0], dtype=np.float32)
        return (
            torch.from_numpy(image),
            torch.from_numpy(label),
            torch.from_numpy(mask),
        )
