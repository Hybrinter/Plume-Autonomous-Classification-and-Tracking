"""Archive index symbols re-exported from the canonical Zenodo source adapter.

Contains:
  - TileRef, TileIndex: one corpus image stem, plus the archive-order index.
  - location_id_of, build_index, iter_stacks: index build and GeoTIFF reads.

The readers and parsers live in
``tools.ml_models.dataset.sources.zenodo.archive``. This module only re-exports
those symbols so study code has a stable local namespace.
"""

from __future__ import annotations

from tools.ml_models.dataset.sources.zenodo.archive import (
    TileIndex,
    TileRef,
    build_index,
    iter_stacks,
    location_id_of,
)

__all__ = [
    "TileIndex",
    "TileRef",
    "build_index",
    "iter_stacks",
    "location_id_of",
]
