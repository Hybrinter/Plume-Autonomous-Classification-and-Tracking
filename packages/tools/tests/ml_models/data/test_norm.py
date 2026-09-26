"""Tests for DN, unit, and per-band z-score recipes."""

import math

import numpy as np
import pytest
from flight.payload.preprocess.normalize import normalize_dn
from tools.ml_models.data.norm import (
    BandStats,
    apply_band_z,
    apply_norm,
    apply_normalize_dn,
    apply_unit,
    fit_band_stats,
)


def test_apply_normalize_dn_maps_full_scale_to_one() -> None:
    """Full-scale 12-bit DN 4095 normalizes to 1.0."""
    planes = np.full((2, 3, 4), 4095.0, dtype=np.float32)
    out = apply_normalize_dn(planes, 12)
    assert out.dtype == np.float32
    assert np.all(out == np.float32(1.0))
    mixed = np.array([[[0.0, 4095.0], [100.0, 5000.0]]], dtype=np.float32)
    assert np.array_equal(apply_normalize_dn(mixed, 12), normalize_dn(mixed, 12))


def test_apply_unit_clips_to_unit_interval() -> None:
    """Values outside [0, 1] clip, and in-range values stay put."""
    out = apply_unit(np.array([-0.5, 0.0, 0.25, 1.0, 1.5], dtype=np.float64))
    assert out.dtype == np.float32
    assert out.tolist() == [0.0, 0.0, 0.25, 1.0, 1.0]


def test_apply_band_z_matches_hand_computed_moments() -> None:
    """Population mean and std match a hand calculation, per channel."""
    stack = np.array([[[1.0, 3.0], [5.0, 7.0]]], dtype=np.float32)
    stats = fit_band_stats(stack)
    assert stats.mean == pytest.approx((4.0,))
    assert stats.std == pytest.approx((math.sqrt(5.0),))
    scaled = apply_band_z(stack, stats)
    expected = (np.array([1.0, 3.0, 5.0, 7.0]) - 4.0) / math.sqrt(5.0)
    np.testing.assert_allclose(scaled.reshape(-1), expected, rtol=1e-5, atol=1e-5)

    two_channel = np.array([[[0.0, 2.0]], [[4.0, 6.0]]], dtype=np.float32)
    per_channel = fit_band_stats(two_channel)
    assert per_channel.mean == pytest.approx((1.0, 5.0))
    assert per_channel.std == pytest.approx((1.0, 1.0))
    batch = np.stack([two_channel, two_channel], axis=0)
    assert fit_band_stats(batch) == per_channel


def test_fit_band_stats_floors_std() -> None:
    """A constant channel has population std 0, stored as 1e-6."""
    constant = np.full((1, 2, 2), 5.0, dtype=np.float32)
    stats = fit_band_stats(constant)
    assert stats.mean == pytest.approx((5.0,))
    assert stats.std == pytest.approx((1e-6,))


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_fit_band_stats_rejects_non_finite_pixels(bad: float) -> None:
    """Non-finite pixels raise ValueError and do not return moments."""
    stack = np.ones((1, 2, 2), dtype=np.float64)
    stack[0, 0, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        fit_band_stats(stack)
    batch = np.ones((2, 1, 2, 2), dtype=np.float32)
    batch[1, 0, 1, 1] = bad
    with pytest.raises(ValueError, match="finite"):
        fit_band_stats(batch)


@pytest.mark.parametrize(
    ("mean", "std"),
    [
        ((math.nan,), (1.0,)),
        ((math.inf,), (1.0,)),
        ((-math.inf,), (1.0,)),
        ((0.0,), (math.nan,)),
        ((0.0,), (math.inf,)),
        ((0.0,), (-math.inf,)),
        ((0.0,), (0.0,)),
        ((0.0,), (-1.0,)),
    ],
)
def test_apply_band_z_rejects_non_finite_stats(
    mean: tuple[float, ...],
    std: tuple[float, ...],
) -> None:
    """NaN, infinity, and non-positive std raise ValueError before scaling."""
    planes = np.ones((1, 2, 2), dtype=np.float32)
    stats = BandStats(mean=mean, std=std)
    with pytest.raises(ValueError, match="finite"):
        apply_band_z(planes, stats)
    batch = np.ones((2, 1, 2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="finite"):
        apply_band_z(batch, stats)


def test_apply_norm_band_z_requires_stats() -> None:
    """band_z without stats raises ValueError."""
    planes = np.zeros((1, 2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="stats"):
        apply_norm(planes, "band_z")


def test_apply_norm_dispatches_and_rejects_bad_bit_depth() -> None:
    """normalize_dn and unit dispatch; bit depth below 1 raises ValueError."""
    planes = np.full((1, 1, 1), 4095.0, dtype=np.float32)
    assert float(apply_norm(planes, "normalize_dn", bit_depth=12)[0, 0, 0]) == pytest.approx(1.0)
    clipped = apply_norm(np.array([2.0, -1.0], dtype=np.float32), "unit")
    assert clipped.tolist() == [1.0, 0.0]
    with pytest.raises(ValueError):
        apply_norm(planes, "normalize_dn", bit_depth=0)
    with pytest.raises(ValueError):
        apply_normalize_dn(planes, 0)
