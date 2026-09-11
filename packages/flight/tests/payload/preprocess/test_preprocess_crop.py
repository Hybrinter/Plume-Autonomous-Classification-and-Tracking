"""Tests for ROI crop, decimate/upsample resampling, and pixel back-projection.

Satisfies: REQ-AIML-PREP-003
"""

from __future__ import annotations

# third-party
import numpy as np

# flight types
from flight.libs.types import Err, FaultCode, Ok

# module under test
from flight.payload.preprocess import (
    RoiTransform,
    crop_and_upsample,
    crop_plane,
    decimate_area,
    decimate_to_size,
    plane_to_tensor_px,
    tensor_to_plane_px,
    upsample,
)


def _planes(c: int = 4, h: int = 16, w: int = 20) -> np.ndarray:
    """Return a (C, H, W) float32 array with distinct per-plane values."""
    arr = np.zeros((c, h, w), dtype=np.float32)
    for i in range(c):
        arr[i, :, :] = float(i + 1)
    return arr


# ---------------------------------------------------------------------------
# crop_plane
# ---------------------------------------------------------------------------


def test_crop_plane_centers_window() -> None:
    """crop_plane returns the requested size centred on center_px."""
    bands = np.arange(4 * 16 * 20, dtype=np.float32).reshape(4, 16, 20)
    result = crop_plane(bands, (10.0, 8.0), (4, 6))
    assert isinstance(result, Ok)
    cropped, origin = result.value
    assert cropped.shape == (4, 4, 6)
    assert origin == (7, 6)  # center - size // 2
    np.testing.assert_array_equal(cropped, bands[:, 6:10, 7:13])


def test_crop_plane_clamps_to_bounds() -> None:
    """A window hanging off the edge is clamped and reports the actual origin."""
    bands = _planes()
    result = crop_plane(bands, (0.0, 0.0), (4, 4))
    assert isinstance(result, Ok)
    _, origin = result.value
    assert origin == (0, 0)


def test_crop_plane_rejects_bad_input() -> None:
    """Non-rank-3 input or a window larger than the plane is Err(FRAME_MALFORMED)."""
    assert isinstance(crop_plane(np.zeros((4, 4), dtype=np.float32), (0, 0), (2, 2)), Err)
    result = crop_plane(_planes(), (8.0, 8.0), (99, 4))
    assert isinstance(result, Err)
    assert result.error == FaultCode.FRAME_MALFORMED


# ---------------------------------------------------------------------------
# decimate_area / decimate_to_size
# ---------------------------------------------------------------------------


def test_decimate_area_box_mean() -> None:
    """factor-2 decimation replaces each 2x2 block with its mean."""
    bands = np.arange(4 * 4 * 4, dtype=np.float32).reshape(4, 4, 4)
    result = decimate_area(bands, 2)
    assert isinstance(result, Ok)
    out = result.value
    assert out.shape == (4, 2, 2)
    expected = bands.reshape(4, 2, 2, 2, 2).mean(axis=(2, 4))
    np.testing.assert_allclose(out, expected)


def test_decimate_area_rejects_non_multiple() -> None:
    """A plane axis that is not a multiple of factor is Err(FRAME_MALFORMED)."""
    result = decimate_area(_planes(h=7), 2)
    assert isinstance(result, Err)
    assert result.error == FaultCode.FRAME_MALFORMED
    assert isinstance(decimate_area(_planes(), 0), Err)


def test_decimate_to_size_returns_transform() -> None:
    """decimate_to_size centre-crops to a factor multiple and reports 1/factor scale."""
    bands = _planes(h=16, w=20)
    result = decimate_to_size(bands, (4, 4))
    assert isinstance(result, Ok)
    tensor, transform = result.value
    assert tensor.shape == (4, 4, 4)
    assert transform.scale_factor == 0.25  # factor 4 = min(16//4, 20//4)
    # Centre crop of (16, 20) to 16x16 drops 2 px on each width side.
    assert transform.crop_origin_px == (2, 0)


def test_decimate_to_size_rejects_oversize() -> None:
    """An output larger than the plane is Err(FRAME_MALFORMED)."""
    assert isinstance(decimate_to_size(_planes(h=4, w=4), (8, 8)), Err)


# ---------------------------------------------------------------------------
# upsample / crop_and_upsample
# ---------------------------------------------------------------------------


def test_upsample_factor_one_returns_copy() -> None:
    """factor 1 returns an Ok float32 copy, not the input object."""
    bands = _planes()
    result = upsample(bands, 1)
    assert isinstance(result, Ok)
    np.testing.assert_array_equal(result.value, bands)
    assert result.value is not bands


def test_upsample_doubles_extent() -> None:
    """factor 2 gives (C, 2H, 2W) output clipped to the input range."""
    bands = np.array([[[0.0, 1.0], [0.0, 0.0]]], dtype=np.float32)
    result = upsample(bands, 2)
    assert isinstance(result, Ok)
    assert result.value.shape == (1, 4, 4)
    assert result.value.min() >= 0.0
    assert result.value.max() <= 1.0


def test_upsample_rejects_bad_factor_and_order() -> None:
    """factor < 1 or order outside [0, 5] returns Err(FRAME_MALFORMED), not a crash."""
    bands = _planes()
    assert isinstance(upsample(bands, 0), Err)
    assert isinstance(upsample(bands, -2), Err)
    bad_order = upsample(bands, 2, order=7)
    assert isinstance(bad_order, Err)
    assert bad_order.error == FaultCode.FRAME_MALFORMED
    assert isinstance(upsample(bands, 2, order=-1), Err)


def test_crop_and_upsample_track_path() -> None:
    """crop_and_upsample returns output_size with scale_factor = upsample_factor."""
    bands = _planes(h=64, w=64)
    result = crop_and_upsample(bands, (32.0, 32.0), (32, 32), 2)
    assert isinstance(result, Ok)
    tensor, transform = result.value
    assert tensor.shape == (4, 32, 32)
    assert transform.scale_factor == 2.0
    assert transform.crop_origin_px == (24, 24)  # 32 - 16


def test_crop_and_upsample_rejects_bad_params() -> None:
    """Invalid factor/order or a non-multiple output size is Err(FRAME_MALFORMED)."""
    bands = _planes(h=64, w=64)
    assert isinstance(crop_and_upsample(bands, (32.0, 32.0), (32, 32), 0), Err)
    assert isinstance(crop_and_upsample(bands, (32.0, 32.0), (32, 32), 2, order=9), Err)
    assert isinstance(crop_and_upsample(bands, (32.0, 32.0), (33, 32), 2), Err)


# ---------------------------------------------------------------------------
# RoiTransform backprojection
# ---------------------------------------------------------------------------


def test_tensor_plane_roundtrip() -> None:
    """tensor_to_plane_px inverts plane_to_tensor_px exactly."""
    transform = RoiTransform(crop_origin_px=(10, 20), scale_factor=0.25)
    plane_px = (30.0, 60.0)
    tensor_px = plane_to_tensor_px(plane_px, transform)
    assert tensor_px == (5.0, 10.0)
    np.testing.assert_allclose(tensor_to_plane_px(tensor_px, transform), plane_px)
