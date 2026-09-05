"""Tests for the causal polynomial encoder-rate estimator."""

import math

from flight.payload.gimbal.rate_fit import fit_rate


def test_short_ring_is_zero() -> None:
    """Fewer than two samples yields y_m = 0."""
    assert fit_rate((), 0.001) == 0.0
    assert fit_rate((0.1,), 0.001) == 0.0


def test_constant_rate_recovers_slope() -> None:
    """A linear ring recovers the constant rate at the newest sample."""
    dt = 0.001
    rate = math.radians(2.0)
    ring = tuple(rate * dt * i for i in range(7))
    y_m = fit_rate(ring, dt, n_omega=7, degree=2)
    assert abs(y_m - rate) < 1e-9


def test_quadratic_matches_instantaneous_not_two_point() -> None:
    """Degree-2 fit at the newest sample is not the two-point chord slope."""
    dt = 0.001
    accel = 2.0
    n = 7
    ring = tuple(0.5 * accel * (dt * i) ** 2 for i in range(n))
    y_m = fit_rate(ring, dt, n_omega=n, degree=2)
    t_new = dt * (n - 1)
    true_rate = accel * t_new
    two_point = (ring[-1] - ring[0]) / ((n - 1) * dt)
    assert abs(y_m - true_rate) < 1e-6
    assert abs(y_m - two_point) > abs(y_m - true_rate)
