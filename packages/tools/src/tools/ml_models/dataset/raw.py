"""Raw tile contract consumed by the dataset build.

Contains:
  - GsdPair: re-export of ``flight.payload.gimbal.footprint.GsdPair``, so a
    tile carries the same metres-per-pixel pair flight computes.
  - BinSpec: an optional GSD bin row.
  - RawTileRef, RawTile: one indexed row and its arrays.
  - Domain: ``dn`` or ``unit``.
  - RawSource: protocol for a forward-only tile stream.

``tile_hw`` is a property of each row. A source may emit more than one spatial
size. ``iter_tiles`` yields rows in ``index`` order in one pass.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from flight.payload.gimbal.footprint import GsdPair

Domain = str

__all__ = ["BinSpec", "Domain", "GsdPair", "RawSource", "RawTile", "RawTileRef"]


@dataclass(frozen=True, slots=True)
class BinSpec:
    """One named GSD bin recorded on the finished dataset.

    Attributes:
        bin_id: Stable bin name.
        lateral_m: Nominal lateral GSD in metres.
        along_m: Nominal along-track GSD in metres.
        elevation_deg: Gimbal elevation for a flight bin. None for a native bin.
    """

    bin_id: str
    lateral_m: float
    along_m: float
    elevation_deg: float | None = None


@dataclass(frozen=True, slots=True)
class RawTileRef:
    """Identity and geometry of one raw tile, without pixel arrays.

    Attributes:
        tile_id: Unique id inside the source.
        group_id: Split key. Rows that share a group stay in one split.
        label: Classification target. Values at or above 0.5 count as positive.
        has_mask: True when the segmentor set includes this row.
        gsd: Pixel GSD at the tile center.
        frame_id: Flight frame id. None when the source has no frame.
        grid_rc: ``(row, col)`` on the flight grid. None when absent.
        bin_id: GSD bin name. Empty when the source has a single geometry.
        theta_g_deg: Gimbal elevation at the shutter, in degrees. None when
            the source does not record one.
        gsd_nominal: True when ``gsd`` comes from nominal orbit geometry
            rather than a measured capture.
    """

    tile_id: str
    group_id: str
    label: float
    has_mask: bool
    gsd: GsdPair
    frame_id: str | None
    grid_rc: tuple[int, int] | None
    bin_id: str
    theta_g_deg: float | None = None
    gsd_nominal: bool = False


@dataclass(frozen=True, slots=True)
class RawTile:
    """One raw tile yielded by a source.

    Attributes:
        ref: Row identity.
        image: np.ndarray[(3, H, W)] in the source domain.
        mask: np.ndarray[(1, H, W)] or None. Present exactly when ``has_mask``.
    """

    ref: RawTileRef
    image: np.ndarray
    mask: np.ndarray | None


@runtime_checkable
class RawSource(Protocol):
    """Forward-only producer of raw tiles.

    Attributes:
        name: Short source name stored on the manifest.
        band_names: Channel names in image order.
        domain: ``dn`` or ``unit``.
        bit_depth: ADC depth used when ``domain`` is ``dn``.
        source_ref: DOI or other provenance string. Empty when the source
            has no external origin.
        extent_m: ``(lateral_m, along_m)`` ground window, or None when every
            tile is the flight 193 by 258 size.
        bins: Bin table copied onto the manifest. Empty when the source has
            no named bins.
    """

    name: str
    band_names: tuple[str, ...]
    domain: str
    bit_depth: int
    source_ref: str
    extent_m: tuple[float, float] | None
    bins: tuple[BinSpec, ...]

    def index(self) -> tuple[RawTileRef, ...]:
        """Return every row in stream order.

        Returns:
            tuple[RawTileRef, ...]: One ref per tile, same order as ``iter_tiles``.
        """
        ...

    def iter_tiles(self) -> Iterator[RawTile]:
        """Yield each tile once, in ``index`` order.

        Returns:
            Iterator[RawTile]: One forward pass.
        """
        ...
