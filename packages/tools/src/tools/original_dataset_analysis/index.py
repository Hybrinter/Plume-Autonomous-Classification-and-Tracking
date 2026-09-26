"""Tile index over the Zenodo image and label archives.

Public names are defined in ``tools.ml_models.data.zenodo``.

Contains:
  - TileRef: one image stem.
  - TileIndex: the corpus index.
  - location_id_of: leading stem token.
  - build_index: stems, presence, and polygons from two tar archives.
  - iter_stacks: one forward pass over requested GeoTIFF members.
"""

from __future__ import annotations

from tools.ml_models.data.zenodo import (
    TileIndex as TileIndex,
)
from tools.ml_models.data.zenodo import (
    TileRef as TileRef,
)
from tools.ml_models.data.zenodo import (
    build_index as build_index,
)
from tools.ml_models.data.zenodo import (
    iter_stacks as iter_stacks,
)
from tools.ml_models.data.zenodo import (
    location_id_of as location_id_of,
)

__all__ = [
    "TileIndex",
    "TileRef",
    "build_index",
    "iter_stacks",
    "location_id_of",
]
