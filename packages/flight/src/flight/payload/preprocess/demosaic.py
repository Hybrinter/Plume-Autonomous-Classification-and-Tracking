"""Accept a prism RGB stack already stored in ``BAND_ORDER``.

The JAI AP-3200T-USB writes three registered planes. There is no color-filter
array to unpack. This step only checks that the stack has one plane per entry
of ``BAND_ORDER``. Cubic upscale runs after this check.

Contains:
  - confirm_planes: accept (3, H, W) float32 in ``BAND_ORDER``, or FRAME_MALFORMED.
"""

from __future__ import annotations

# third-party
import numpy as np

# internal
from flight.libs.types import BAND_ORDER, Err, FaultCode, Ok, Result


def confirm_planes(planes: np.ndarray) -> Result[np.ndarray, FaultCode]:
    """Return the RGB stack when it has one plane per ``BAND_ORDER`` entry.

    Inputs:
        planes (np.ndarray): Candidate (3, H, W) stack.

    Outputs:
        Result[np.ndarray, FaultCode]:
            Ok(float32 (3, H, W)) on success.
            Err(FRAME_MALFORMED) when rank or channel count is wrong.
    """
    if planes.ndim != 3 or planes.shape[0] != len(BAND_ORDER):
        return Err(FaultCode.FRAME_MALFORMED)
    return Ok(planes.astype(np.float32, copy=False))
