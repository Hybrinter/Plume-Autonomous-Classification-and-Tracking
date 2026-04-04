"""
pact.preprocessing.band_select — Sentinel-2 band selection for PACT inference input.

Satisfies: REQ-AIML-PREP-001, REQ-AIML-IMAG-001

Selects bands B2 (490 nm), B3 (560 nm), B4 (665 nm), and B8 (842 nm) from a raw
multispectral frame. These four VNIR bands are the input to the U-Net/ResNet-34 model.

Note on indexing: The raw camera produces frames with C_total bands in a fixed ordering
defined by the sensor configuration. BAND_INDICES maps the logical band name to its
zero-based index in the raw (C_total, H, W) array. The current values (0,1,2,3) assume
the camera has already been configured to deliver exactly these four bands in order.
If the sensor delivers a superset of bands, update BAND_INDICES accordingly.
TODO: confirm band ordering with FLIR Blackfly S multispectral filter wheel configuration.
"""

from __future__ import annotations

# stdlib
from typing import Final

# third-party
import numpy as np


# Maps logical Sentinel-2 band names to zero-based indices in the raw sensor output.
# Sentinel-2 13-band canonical ordering: B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B10,B11,B12
# Indices 1,2,3,7 → B2 (490 nm), B3 (560 nm), B4 (665 nm), B8 (842 nm).
# These match the indices used in HsgAimlDataset.__getitem__().
BAND_INDICES: Final[dict[str, int]] = {
    "B2": 0,   # 490 nm — blue
    "B3": 1,   # 560 nm — green
    "B4": 2,   # 665 nm — red
    "B8": 3,   # 842 nm — near-infrared (NIR)
}


def select_bands(
    raw: np.ndarray,              # (C_total, H, W) float32
    band_names: tuple[str, ...],
) -> np.ndarray:                  # (len(band_names), H, W) float32
    """Select named bands from a raw multispectral array.

    Looks up each name in BAND_INDICES and gathers the corresponding slices.
    The output channel order matches the order of band_names.

    Args:
        raw:        Raw multispectral frame. Shape (C_total, H, W), dtype float32.
                    C_total must be >= max(BAND_INDICES.values()) + 1.
        band_names: Ordered tuple of band names to select, e.g. ("B2", "B3", "B4", "B8").
                    All names must be keys in BAND_INDICES.

    Returns:
        Selected bands as np.ndarray of shape (len(band_names), H, W), float32.

    Raises:
        KeyError:      If any name in band_names is not in BAND_INDICES.
        IndexError:    If any resolved index is out of range for raw.shape[0].
    """
    indices: list[int] = [BAND_INDICES[name] for name in band_names]
    return raw[indices, :, :]  # np.ndarray[float32, (len(band_names), H, W)]
