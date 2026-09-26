"""Stack a prism camera buffer into channel-major planes.

The AP-3200T-USB delivers three registered sensors. This module accepts a
channel-major ``(3, H, W)`` buffer or a channel-last ``(H, W, 3)`` buffer and
returns float32 ``(3, H, W)``. A 2-D buffer is malformed.

Satisfies: REQ-AIML-PREP-001, REQ-AIML-IMAG-001.
"""

from __future__ import annotations

# third-party
import numpy as np

# internal
from flight.libs.types import Err, FaultCode, Ok, Result


def stack_channels(buffer: np.ndarray) -> Result[np.ndarray, FaultCode]:
    """Return a float32 ``(3, H, W)`` stack from a prism camera buffer.

    Args:
        buffer: ``(3, H, W)`` or ``(H, W, 3)`` numeric array. Channel count is 3.

    Returns:
        Ok(np.ndarray[float32, (3, H, W)]) on success.
        Err(FaultCode.FRAME_MALFORMED) when the buffer is not a 3-channel volume.
    """
    if buffer.ndim == 3 and buffer.shape[0] == 3:
        planes = buffer
    elif buffer.ndim == 3 and buffer.shape[-1] == 3:
        planes = np.moveaxis(buffer, -1, 0)
    else:
        return Err(FaultCode.FRAME_MALFORMED)
    return Ok(np.asarray(planes, dtype=np.float32))
