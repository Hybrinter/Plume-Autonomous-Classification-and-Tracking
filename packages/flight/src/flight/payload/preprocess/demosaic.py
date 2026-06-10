"""2x2 CFA separation: raw mosaic plane <-> registered band planes.

The 2x2 tile repeats across the sensor; band plane k is the stride-2 sample of
row-major cell k. Planes are half the mosaic resolution and spatially registered to
each other (no interpolation -- plane co-registration error is half a mosaic pixel,
absorbed into the pointing budget). Band NAMES are assigned by SensorConfig.mosaic_layout
in the same row-major cell order; this module is layout-agnostic.

interleave_bands is the exact inverse of separate_bands, used by the sim scene renderer
and round-trip tests to reconstruct a mosaic from its band planes.

Satisfies: REQ-AIML-PREP-001, REQ-AIML-IMAG-001.
"""

from __future__ import annotations

# stdlib
from typing import Final

# third-party
import numpy as np

# internal
from flight.libs.types import Err, FaultCode, Ok, Result

# Row-major (row_offset, col_offset) of each 2x2 cell; plane order follows this.
# Cell (0,0) -> plane 0 (BLUE), (0,1) -> plane 1 (GREEN), (1,0) -> plane 2 (RED),
# (1,1) -> plane 3 (NIR) when using the default SensorConfig.mosaic_layout.
CELL_OFFSETS: Final[tuple[tuple[int, int], ...]] = ((0, 0), (0, 1), (1, 0), (1, 1))


def separate_bands(mosaic: np.ndarray) -> Result[np.ndarray, FaultCode]:
    """Split a (H, W) mosaic plane into (4, H/2, W/2) float32 band planes.

    Each band plane k is extracted by striding over the mosaic at step 2,
    starting at the row-major cell offset given by CELL_OFFSETS[k]. This
    gives four spatially-registered half-resolution planes with no interpolation.

    Args:
        mosaic: np.ndarray of shape (H, W) with even H and W; any numeric dtype.
                float32 input is used as-is; other dtypes are cast to float32 in
                the output stack.

    Returns:
        Ok(np.ndarray[float32, (4, H/2, W/2)]) on success.
        Err(FaultCode.FRAME_MALFORMED) if mosaic is not 2-D or has an odd dimension.

    Notes:
        The caller is responsible for ensuring the mosaic matches the sensor geometry
        declared in SensorConfig (H == height_px, W == width_px). This function only
        checks structural validity (rank and parity), not absolute size.
    """
    if mosaic.ndim != 2 or mosaic.shape[0] % 2 != 0 or mosaic.shape[1] % 2 != 0:
        return Err(FaultCode.FRAME_MALFORMED)
    planes = np.stack([mosaic[r::2, c::2] for r, c in CELL_OFFSETS]).astype(
        np.float32
    )  # np.ndarray[float32, (4, H/2, W/2)]
    return Ok(planes)


def interleave_bands(planes: np.ndarray) -> Result[np.ndarray, FaultCode]:
    """Rebuild the (H, W) mosaic from (4, h, w) band planes (exact inverse of separate_bands).

    Scatters each band plane back into its 2x2 cell position using the same
    CELL_OFFSETS stride pattern used by separate_bands. The output dtype matches
    the input planes dtype.

    Args:
        planes: np.ndarray of shape (4, h, w); any dtype. h and w become H/2 and W/2
                of the reconstructed mosaic.

    Returns:
        Ok(np.ndarray[planes.dtype, (2*h, 2*w)]) on success.
        Err(FaultCode.FRAME_MALFORMED) if planes is not rank-3 or has fewer/more than 4 planes.

    Notes:
        Used by the sim scene renderer to build raw mosaic frames from per-band signal
        maps, and by round-trip unit tests to verify that separate_bands is lossless.
        The four CELL_OFFSETS partition the 2x2 tile, so every mosaic cell is written
        exactly once and the empty allocation is fully overwritten.
    """
    if planes.ndim != 3 or planes.shape[0] != 4:
        return Err(FaultCode.FRAME_MALFORMED)
    h, w = planes.shape[1], planes.shape[2]
    mosaic = np.empty((2 * h, 2 * w), dtype=planes.dtype)  # np.ndarray[planes.dtype, (2*h, 2*w)]
    for k, (r, c) in enumerate(CELL_OFFSETS):
        mosaic[r::2, c::2] = planes[k]
    return Ok(mosaic)
