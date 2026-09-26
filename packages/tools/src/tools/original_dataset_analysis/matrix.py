"""Native-resolution band matrix.

Public names are defined in ``tools.ml_models.data.matrix``.

Contains:
  - Cell: one task, subset name, and side.
  - native_cells: the 12-band set, RGB, and leave-one-out.
  - gsd_cells: the 12-band set and RGB at every legal side.
  - require_complete: refuse a table that omits a planned cell.
"""

from __future__ import annotations

from tools.ml_models.data.matrix import (
    Cell as Cell,
)
from tools.ml_models.data.matrix import (
    gsd_cells as gsd_cells,
)
from tools.ml_models.data.matrix import (
    native_cells as native_cells,
)
from tools.ml_models.data.matrix import (
    require_complete as require_complete,
)

__all__ = [
    "Cell",
    "gsd_cells",
    "native_cells",
    "require_complete",
]
