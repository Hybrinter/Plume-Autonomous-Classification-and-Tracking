"""Unit-interval conversion and the model GSD encoding.

Contains:
  - to_unit: flight DN through ``normalize_dn``, or a clip of an existing unit image.
  - quantize_unit: store a unit image as uint16.
  - to_model_gsd: ``ln(gsd / gsd_reference_m)`` per component.

Stored images use the uint16 grid ``round(unit * 65535)``. Reading divides by
65535. Storage is a lossy quantization: the absolute error on a stored unit
value is at most 0.5 / 65535, so normalized float values are not exactly
preserved. A 12-bit integer DN can still be recovered exactly by rounding the
stored value back onto the DN grid.
"""

from __future__ import annotations

import numpy as np
from flight.libs.types import Err
from flight.payload.gimbal.footprint import to_model_gsd as _flight_to_model_gsd
from flight.payload.preprocess.normalize import normalize_dn

IMAGE_SCALE = 65535


def to_unit(image: np.ndarray, domain: str, bit_depth: int) -> np.ndarray:
    """Map a raw tile image into the unit interval.

    Args:
        image: np.ndarray[(C, H, W)] raw planes.
        domain: ``dn`` or ``unit``.
        bit_depth: ADC depth. Used only when ``domain`` is ``dn``.

    Returns:
        np.ndarray[float32, (C, H, W)] with values in ``[0, 1]``.

    Raises:
        ValueError: If ``domain`` is unknown or ``bit_depth`` is below 1.
    """
    if domain == "dn":
        if bit_depth < 1:
            raise ValueError(f"bit_depth must be >= 1; got {bit_depth}")
        return normalize_dn(np.asarray(image, dtype=np.float32), bit_depth)
    if domain == "unit":
        return np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    raise ValueError(f"unknown domain {domain!r}")


def quantize_unit(image: np.ndarray) -> np.ndarray:
    """Pack a unit image onto the uint16 grid.

    Args:
        image: np.ndarray[float32, (C, H, W)] in ``[0, 1]``.

    Returns:
        np.ndarray[uint16, (C, H, W)]: ``round(clip(image) * 65535)``.
    """
    clipped = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    scaled = np.rint(clipped * np.float32(IMAGE_SCALE))
    return np.asarray(scaled.astype(np.uint16))


def dequantize_unit(image: np.ndarray) -> np.ndarray:
    """Map a stored uint16 image back to the unit interval.

    Args:
        image: np.ndarray[uint16, (C, H, W)].

    Returns:
        np.ndarray[float32, (C, H, W)]: ``image / 65535``.
    """
    return np.asarray(image, dtype=np.float32) / np.float32(IMAGE_SCALE)


def to_model_gsd(gsd_m: np.ndarray, reference_m: float) -> np.ndarray:
    """Encode metres of GSD as a log ratio to the flight reference.

    Delegates to ``flight.payload.gimbal.footprint.to_model_gsd`` so the
    training encoding and the flight encoding share one formula.

    Args:
        gsd_m: np.ndarray[float32, (..., 2)] lateral then along-track metres.
        reference_m: Positive reference GSD in metres.

    Returns:
        np.ndarray[float32, (..., 2)]: ``ln(gsd_m / reference_m)``.

    Raises:
        ValueError: If the reference is not finite and greater than 0, or the
            array lacks a trailing length-2 axis of finite positive metres.
    """
    encoded = _flight_to_model_gsd(gsd_m, reference_m)
    if isinstance(encoded, Err):
        raise ValueError("gsd metres must be finite, > 0, with a trailing length-2 axis")
    return encoded.value
