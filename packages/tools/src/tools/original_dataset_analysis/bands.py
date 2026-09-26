"""Sentinel-2 band identity and subset selection for Zenodo 4250706.

Public names are defined in ``tools.ml_models.data.bands``.

Contains:
  - BandOrder: verified file order of Sentinel-2 ids.
  - BandSpec: named subset request.
  - BandSubset: ids and file indices for one run.
  - coerce_descriptions: fill the corpus order when descriptions are empty.
  - verify_band_order: descriptions to a BandOrder.
  - resolve_subset: a BandSpec against a verified order.
"""

from __future__ import annotations

from tools.ml_models.data.bands import (
    BandOrder as BandOrder,
)
from tools.ml_models.data.bands import (
    BandSpec as BandSpec,
)
from tools.ml_models.data.bands import (
    BandSubset as BandSubset,
)
from tools.ml_models.data.bands import (
    coerce_descriptions as coerce_descriptions,
)
from tools.ml_models.data.bands import (
    resolve_subset as resolve_subset,
)
from tools.ml_models.data.bands import (
    verify_band_order as verify_band_order,
)

__all__ = [
    "BandOrder",
    "BandSpec",
    "BandSubset",
    "coerce_descriptions",
    "resolve_subset",
    "verify_band_order",
]
