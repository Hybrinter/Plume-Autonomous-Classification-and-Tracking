"""2x2 CFA separation: raw mosaic plane <-> registered band planes.

The 2x2 tile repeats across the sensor; band plane k is the stride-2 sample of
row-major cell k. Planes are half the mosaic resolution and spatially registered to
each other (no interpolation -- plane co-registration error is half a mosaic pixel,
which at the 150 mm / 3.45 um flight geometry is 0.00066 deg, far inside the 0.1 deg
pointing budget). Band NAMES are assigned by SensorConfig.mosaic_layout in the same
row-major cell order; this module is layout-agnostic.

CFA phase: the flight CFA is an overlay bonded in front of the IMX264, not a
lithographic pattern, so the first COMPLETE 2x2 tile may not start at mosaic pixel
(0, 0). cfa_phase = (row, col) in {0, 1}^2 names the mosaic pixel where the first
complete tile begins. separate_bands discards the partial leading row/column (and the
trailing row/column that becomes partial) so that plane k is always cell k of a complete
tile. With the default (0, 0) the full mosaic is used and behaviour is unchanged.

interleave_bands is the exact inverse of separate_bands for the same cfa_phase, used by
the sim scene renderer and round-trip tests to reconstruct a mosaic from its band
planes. For a nonzero phase it emits zero-filled partial-tile borders so that
separate_bands(interleave_bands(planes, phase), phase) == planes.

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

_DEFAULT_PHASE: Final[tuple[int, int]] = (0, 0)


def _phase_valid(cfa_phase: tuple[int, int]) -> bool:
    """True when cfa_phase is a 2-element sequence of values each in {0, 1}.

    Inputs:
        cfa_phase (tuple[int, int]): (row, col) phase.

    Outputs:
        bool: Validity of the phase.
    """
    try:
        return len(cfa_phase) == 2 and cfa_phase[0] in (0, 1) and cfa_phase[1] in (0, 1)
    except TypeError:
        return False


def separate_bands(
    mosaic: np.ndarray,
    cfa_phase: tuple[int, int] = _DEFAULT_PHASE,
) -> Result[np.ndarray, FaultCode]:
    """Split a (H, W) mosaic plane into (4, h, w) float32 band planes.

    Each band plane k is extracted by striding over the mosaic at step 2, starting at
    cfa_phase + CELL_OFFSETS[k]. This gives four spatially-registered half-resolution
    planes with no interpolation.

    Inputs:
        mosaic (np.ndarray[*, (H, W)]): Mosaic with even H and W; any numeric dtype.
            float32 input is used as-is; other dtypes are cast to float32 in the
            output stack.
        cfa_phase (tuple[int, int]): (row, col) of the first complete 2x2 tile, each
            in {0, 1}. Default (0, 0).

    Outputs:
        Result[np.ndarray, FaultCode]:
            Ok(np.ndarray[float32, (4, h, w)]) with h = (H - 2*row) / 2 and
                w = (W - 2*col) / 2: for a nonzero phase the leading partial row/column
                and the trailing row/column are dropped so the plane covers complete
                tiles only.
            Err(FaultCode.FRAME_MALFORMED) if mosaic is not 2-D, has an odd dimension,
                is smaller than one complete tile after the phase crop, or cfa_phase is
                outside {0, 1}^2.

    Notes:
        The caller is responsible for ensuring the mosaic matches the sensor geometry
        declared in SensorConfig (H == height_px, W == width_px). This function only
        checks structural validity (rank and parity), not absolute size. The pointing
        code treats the plane centre as the boresight; a nonzero phase shifts that
        centre by phase/2 mosaic pixels, which is negligible against the pointing budget.
    """
    if not _phase_valid(cfa_phase):
        return Err(FaultCode.FRAME_MALFORMED)
    if mosaic.ndim != 2 or mosaic.shape[0] % 2 != 0 or mosaic.shape[1] % 2 != 0:
        return Err(FaultCode.FRAME_MALFORMED)
    r0, c0 = cfa_phase
    h_full = mosaic.shape[0] - 2 * r0
    w_full = mosaic.shape[1] - 2 * c0
    if h_full < 2 or w_full < 2:
        return Err(FaultCode.FRAME_MALFORMED)
    tiles = mosaic[r0 : r0 + h_full, c0 : c0 + w_full]  # np.ndarray[*, (H-2r, W-2c)]
    planes = np.stack([tiles[r::2, c::2] for r, c in CELL_OFFSETS]).astype(
        np.float32
    )  # np.ndarray[float32, (4, h, w)]
    return Ok(planes)


def interleave_bands(
    planes: np.ndarray,
    cfa_phase: tuple[int, int] = _DEFAULT_PHASE,
) -> Result[np.ndarray, FaultCode]:
    """Rebuild the mosaic from (4, h, w) band planes (exact inverse of separate_bands).

    Scatters each band plane back into its 2x2 cell position using the same
    CELL_OFFSETS stride pattern used by separate_bands, offset by cfa_phase. The output
    dtype matches the input planes dtype.

    Inputs:
        planes (np.ndarray[*, (4, h, w)]): Band planes in CELL_OFFSETS order; any dtype.
        cfa_phase (tuple[int, int]): (row, col) of the first complete tile, each in
            {0, 1}. Default (0, 0).

    Outputs:
        Result[np.ndarray, FaultCode]:
            Ok(np.ndarray[planes.dtype, (2h + 2*row, 2w + 2*col)]). For a nonzero phase
                the leading `row`/`col` and trailing `row`/`col` lines are the partial
                tiles at the sensor edge and are zero-filled.
            Err(FaultCode.FRAME_MALFORMED) if planes is not rank-3, has fewer/more than
                4 planes, or cfa_phase is outside {0, 1}^2.

    Notes:
        Used by the sim scene renderer to build raw mosaic frames from per-band signal
        maps, and by round-trip unit tests to verify that separate_bands is lossless.
        The four CELL_OFFSETS partition the 2x2 tile, so every complete-tile cell is
        written exactly once.
    """
    if not _phase_valid(cfa_phase):
        return Err(FaultCode.FRAME_MALFORMED)
    if planes.ndim != 3 or planes.shape[0] != 4:
        return Err(FaultCode.FRAME_MALFORMED)
    r0, c0 = cfa_phase
    h, w = planes.shape[1], planes.shape[2]
    mosaic = np.zeros(
        (2 * h + 2 * r0, 2 * w + 2 * c0), dtype=planes.dtype
    )  # np.ndarray[planes.dtype, (2h+2r, 2w+2c)]
    for k, (r, c) in enumerate(CELL_OFFSETS):
        mosaic[r0 + r : r0 + 2 * h : 2, c0 + c : c0 + 2 * w : 2] = planes[k]
    return Ok(mosaic)
