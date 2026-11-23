"""Band-order helpers re-exported from the canonical Zenodo source adapter.

Contains:
  - BandOrder, BandSpec, BandSubset: verified corpus band layouts.
  - coerce_descriptions, verify_band_order, resolve_subset: band plumbing.

The archive parsing and validation live in
``tools.ml_models.dataset.sources.zenodo.bands``. This module only re-exports
those symbols so study code has a stable local namespace.
"""

from __future__ import annotations

from tools.ml_models.dataset.sources.zenodo.bands import (
    ZENODO_BAND_IDS,
    BandOrder,
    BandSpec,
    BandSubset,
    coerce_descriptions,
    resolve_subset,
    verify_band_order,
)

__all__ = [
    "ZENODO_BAND_IDS",
    "BandOrder",
    "BandSpec",
    "BandSubset",
    "coerce_descriptions",
    "resolve_subset",
    "verify_band_order",
]
