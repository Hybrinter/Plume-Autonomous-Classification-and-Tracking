"""Location-grouped train, validation, and test assignment.

Public names are defined in ``tools.ml_models.data.zenodo``.

Contains:
  - SplitRecipe: seed and site fractions.
  - LocationSplit: the site assignment and the stems in each split.
  - assign_location_splits: one split name per location id.
"""

from __future__ import annotations

from tools.ml_models.data.zenodo import (
    LocationSplit as LocationSplit,
)
from tools.ml_models.data.zenodo import (
    SplitRecipe as SplitRecipe,
)
from tools.ml_models.data.zenodo import (
    assign_location_splits as assign_location_splits,
)

__all__ = [
    "LocationSplit",
    "SplitRecipe",
    "assign_location_splits",
]
