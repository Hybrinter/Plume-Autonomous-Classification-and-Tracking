"""DN -> [0, 1] normalization for calibrated band planes.

normalized = clip(dn / (2**bit_depth - 1), 0, 1). This is the reflectance-like domain
the quality thresholds and the model input contract assume (spec Section 4: the model
manifest's input domain is exactly this function's output). Clipping bounds calibration
under/overshoot; saturation detection still works because saturated pixels land at 1.0.

normalize_dn: scale calibrated DN values by ADC full scale and clip to [0, 1] float32.

Satisfies: REQ-AIML-PREP-002.
"""

from __future__ import annotations

# third-party
import numpy as np


def normalize_dn(planes: np.ndarray, bit_depth: int) -> np.ndarray:
    """Normalize calibrated DN band planes to [0, 1] float32 by ADC full scale.

    Divides every element by (2**bit_depth - 1) then clips to [0, 1]. Values below
    zero arise from dark-subtraction overshoot and are clipped to 0.0; values above
    full scale arise from saturation/calibration artefacts and are clipped to 1.0.
    Saturation detection downstream still works because saturated pixels land at 1.0.

    Args:
        planes: np.ndarray[float32, (C, H, W)] calibrated DN values.
        bit_depth: ADC bit depth; full scale is 2**bit_depth - 1 (e.g. 4095 for 12-bit).

    Returns:
        np.ndarray[float32, (C, H, W)] with all values in [0, 1].

    Notes:
        The output dtype is always float32 regardless of the input dtype.
        This function is a pure transformation: no I/O, no global state.
    """
    full_scale = float(2**bit_depth - 1)
    return np.clip(planes / full_scale, 0.0, 1.0).astype(np.float32)
