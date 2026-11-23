"""The model GSD encoding.

Contains:
  - to_model_gsd: ``ln(gsd / gsd_reference_m)`` per component.

Finished datasets store float32 unit pixels; no pixel normalization or
quantization happens here.
"""

from __future__ import annotations

import numpy as np
from flight.libs.types import Err
from flight.payload.gimbal.footprint import to_model_gsd as _flight_to_model_gsd


def to_model_gsd(gsd_m: np.ndarray, reference_m: float) -> np.ndarray:
    """Encode metres of GSD as a log ratio to the flight reference.

    Args:
        gsd_m: np.ndarray[float32, (..., 2)] lateral then along-track metres.
        reference_m: Positive reference GSD in metres.

    Returns:
        np.ndarray[float32, (..., 2)]: ``ln(gsd_m / reference_m)``.

    Raises:
        ValueError: If the reference or any GSD component is not finite and
            greater than 0.
    """
    result = _flight_to_model_gsd(np.asarray(gsd_m), reference_m)
    if isinstance(result, Err):
        raise ValueError(f"flight GSD encoding rejected input: {result.error}")
    return result.value
