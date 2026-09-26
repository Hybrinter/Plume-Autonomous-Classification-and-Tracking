"""Tile samples for the band study.

Contains:
  - TileReader: reads one native stack.
  - TileSample: image, label, mask, and annotation flag.
  - StudyDataset: selected bands, a legal side, and frozen moments.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from tools.ml_models.data.bands import BandSubset
from tools.ml_models.data.grid import coarsen, rasterize_mask
from tools.ml_models.data.moments import BandStats, apply_band_stats, check_band_stats
from tools.ml_models.data.zenodo import SidePack, TileRef

TileReader = Callable[[TileRef], np.ndarray]
TileSample = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class StudyDataset(Dataset[TileSample]):
    """One tile at a fixed band subset and ground-sample side.

    Each item is ``(image, label, mask, annotated)``. ``label`` and
    ``annotated`` have shape ``(1,)``. ``mask`` has shape ``(1, side, side)``.
    ``annotated`` is 1 when ``polygons`` is not ``None`` and 0 when the tile
    has no annotation file. A zero mask with ``annotated`` 0 is not an empty
    target.
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
        cached_masks: np.ndarray | None = None,
        mask_rows: Mapping[str, int] | None = None,
    ) -> None:
        """Store the tiles and the transforms applied on each read.

        Args:
            tiles: Tiles in this split.
            reader: Returns a native ``(bands, 120, 120)`` stack in file order.
            subset: Channels kept, in file order.
            stats: Frozen per-channel moments for ``subset``.
            side_px: Legal output side.
            mask_rule: ``touch`` or ``half``.
            cached_masks: Optional ``(N, side, side)`` masks. ``None`` rasterizes.
            mask_rows: Stem to row in ``cached_masks``.

        Raises:
            ValueError: If stats do not match the subset width.
        """
        check_band_stats(stats, len(subset.indices))
        self._tiles = tuple(tiles)
        self._reader = reader
        self._subset = subset
        self._stats = stats
        self._side_px = side_px
        self._mask_rule = mask_rule
        self._cached_masks = cached_masks
        self._mask_rows = mask_rows
        self._pack: SidePack | None = None
        self._rows = np.empty(0, dtype=np.int32)
        self._channels: tuple[int, ...] = ()

    @classmethod
    def from_pack(
        cls,
        pack: SidePack,
        rows: np.ndarray,
        channels: Sequence[int],
        stats: BandStats,
    ) -> StudyDataset:
        """Read a prepared side pack. No resampling happens in ``__getitem__``.

        Args:
            pack: One open ground-sample size.
            rows: Int32 pack rows for this split.
            channels: Channel index into the pack, in file order.
            stats: Frozen moments for those channels.

        Returns:
            StudyDataset: Samples drawn from ``pack``.

        Raises:
            ValueError: If the moments do not match ``channels``.
        """
        check_band_stats(stats, len(channels))
        dataset = cls.__new__(cls)
        dataset._pack = pack
        dataset._rows = np.asarray(rows, dtype=np.int32)
        dataset._channels = tuple(int(channel) for channel in channels)
        dataset._stats = stats
        dataset._tiles = ()
        dataset._side_px = pack.side_px
        dataset._mask_rule = "half"
        dataset._cached_masks = None
        dataset._mask_rows = None
        return dataset

    def __len__(self) -> int:
        """Return the number of tiles."""
        if self._pack is not None:
            return int(self._rows.shape[0])
        return len(self._tiles)

    def _pack_item(self, index: int) -> TileSample:
        """Return one sample from the prepared pack."""
        pack = self._pack
        if pack is None:
            raise ValueError("pack dataset has no pack")
        row = int(self._rows[index])
        selected = np.asarray(pack.images[row][list(self._channels)], dtype=np.float32)
        image = apply_band_stats(selected, self._stats)
        mask = np.asarray(pack.masks[row], dtype=np.float32).reshape(1, pack.side_px, pack.side_px)
        label = np.array([float(pack.positive[row])], dtype=np.float32)
        annotated = np.array([float(pack.annotated[row])], dtype=np.float32)
        return (
            torch.from_numpy(image),
            torch.from_numpy(label),
            torch.from_numpy(mask),
            torch.from_numpy(annotated),
        )

    def __getitem__(self, index: int) -> TileSample:
        """Return one normalized image, its presence label, its mask, and a flag.

        Args:
            index: Tile position.

        Returns:
            tuple: Image ``(C, side, side)``, label ``(1,)``, mask
            ``(1, side, side)``, and annotation flag ``(1,)``.
        """
        if self._pack is not None:
            return self._pack_item(index)
        tile = self._tiles[index]
        native = np.asarray(self._reader(tile), dtype=np.float32)
        selected = native[list(self._subset.indices)]
        image = apply_band_stats(coarsen(selected, self._side_px), self._stats)
        if tile.polygons is None:
            mask = np.zeros((1, self._side_px, self._side_px), dtype=np.float32)
            annotated = np.array([0.0], dtype=np.float32)
        elif (
            self._cached_masks is not None
            and self._mask_rows is not None
            and self._cached_masks.shape[-1] == self._side_px
        ):
            plane = np.asarray(self._cached_masks[self._mask_rows[tile.stem]], dtype=np.float32)
            mask = plane.reshape(1, self._side_px, self._side_px)
            annotated = np.array([1.0], dtype=np.float32)
        else:
            mask = rasterize_mask(tile.polygons, self._side_px, rule=self._mask_rule)
            annotated = np.array([1.0], dtype=np.float32)
        label = np.array([1.0 if tile.positive else 0.0], dtype=np.float32)
        return (
            torch.from_numpy(image),
            torch.from_numpy(label),
            torch.from_numpy(mask),
            torch.from_numpy(annotated),
        )
