"""Cubic upscale of an RGB stack after the planes are confirmed.

``upsample_planes`` enlarges height and width by an integer factor. Channel
count stays ``len(BAND_ORDER)``. Factor 1 returns the input. Factor 2 uses a
separable kernel: order 3 is a Keys cubic, order 1 is a linear midpoint.
Any other factor uses ``scipy.ndimage.zoom``.

Contains:
  - upsample_planes: zoom (3, H, W) to (3, factor*H, factor*W).
"""

from __future__ import annotations

# third-party
import numpy as np
from scipy.ndimage import zoom

# internal
from flight.libs.types import BAND_ORDER, Err, FaultCode, Ok, Result

_ORDERS: frozenset[int] = frozenset({1, 3})


def _double_axis(stack: np.ndarray, axis: int, order: int) -> np.ndarray:
    """Double one axis of a (3, H, W) stack with a linear or Keys cubic kernel."""
    length = stack.shape[axis]
    pad = [(0, 0), (0, 0), (0, 0)]
    pad[axis] = (1, 2)
    padded = np.pad(stack, pad, mode="edge")

    def _take(start: int, stop: int) -> np.ndarray:
        index: list[slice] = [slice(None), slice(None), slice(None)]
        index[axis] = slice(start, stop)
        return padded[tuple(index)]

    left = _take(1, length + 1)
    right = _take(2, length + 2)
    if order == 1:
        mid = 0.5 * (left + right)
    else:
        mid = (-_take(0, length) + 9.0 * left + 9.0 * right - _take(3, length + 3)) / 16.0
    out_shape = [stack.shape[0], stack.shape[1], stack.shape[2]]
    out_shape[axis] = length * 2
    out = np.empty(out_shape, dtype=np.float32)
    even: list[slice] = [slice(None), slice(None), slice(None)]
    odd: list[slice] = [slice(None), slice(None), slice(None)]
    even[axis] = slice(0, None, 2)
    odd[axis] = slice(1, None, 2)
    out[tuple(even)] = stack
    out[tuple(odd)] = mid
    return out


def upsample_planes(
    planes: np.ndarray,
    factor: int,
    order: int = 3,
) -> Result[np.ndarray, FaultCode]:
    """Upscale each RGB plane by ``factor`` with interpolation order ``order``.

    Inputs:
        planes (np.ndarray[float32, (3, H, W)]): Planes in ``BAND_ORDER``.
        factor (int): Integer scale. 1 leaves the array unchanged.
        order (int): 1 linear or 3 cubic.

    Outputs:
        Result[np.ndarray, FaultCode]:
            Ok(float32 (3, factor*H, factor*W)).
            Err(FRAME_MALFORMED) when factor < 1, order is not 1 or 3, or the
            input is not a 3-plane stack.
    """
    if planes.ndim != 3 or planes.shape[0] != len(BAND_ORDER) or factor < 1 or order not in _ORDERS:
        return Err(FaultCode.FRAME_MALFORMED)
    values = planes.astype(np.float32, copy=False)
    if factor == 1:
        return Ok(values)
    if factor == 2:
        doubled = _double_axis(values, axis=1, order=order)
        return Ok(_double_axis(doubled, axis=2, order=order))
    zoomed = zoom(values, (1.0, float(factor), float(factor)), order=order)
    return Ok(np.asarray(zoomed, dtype=np.float32))
