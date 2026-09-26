"""Fixed 1.2 km tile grid, reflectance coarsening, and mask rasterization.

Public names are defined in ``tools.ml_models.data.grid``.

Contains:
  - LEGAL_SIDES: pixel sides that divide 1200 m into 10, 15, 20, 30, or 40 m.
  - gsd_m: ground sample distance for a legal side.
  - coarsen: area-weighted resample from the native 120 px grid.
  - rasterize_mask: percentage polygons onto a legal side.
"""

from __future__ import annotations

from tools.ml_models.data.grid import (
    EXTENT_M as EXTENT_M,
)
from tools.ml_models.data.grid import (
    LEGAL_SIDES as LEGAL_SIDES,
)
from tools.ml_models.data.grid import (
    NATIVE_SIDE as NATIVE_SIDE,
)
from tools.ml_models.data.grid import (
    coarsen as coarsen,
)
from tools.ml_models.data.grid import (
    gsd_m as gsd_m,
)
from tools.ml_models.data.grid import (
    rasterize_mask as rasterize_mask,
)

__all__ = [
    "EXTENT_M",
    "LEGAL_SIDES",
    "NATIVE_SIDE",
    "coarsen",
    "gsd_m",
    "rasterize_mask",
]
