"""Band order lives in ``flight.libs.types.BAND_ORDER``.

Preprocess does not reorder channels. The driver stacks BLUE, GREEN, RED in
that tuple, and the model reads the same tuple.

Contains:
  - canonical_band_names: the ``BAND_ORDER`` names as strings.
"""

from __future__ import annotations

# internal
from flight.libs.types import BAND_ORDER


def canonical_band_names() -> tuple[str, ...]:
    """Return the only band order, as strings.

    Outputs:
        tuple[str, ...]: ``("BLUE", "GREEN", "RED")``.
    """
    return tuple(band.value for band in BAND_ORDER)
