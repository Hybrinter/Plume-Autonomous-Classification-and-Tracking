"""Causal polynomial rate estimator on a uniform encoder ring (pure).

Fits a degree-d polynomial on the last n samples with the newest sample at tau=0
and returns the first derivative at that sample. This is a causal Savitzky-Golay /
least-squares differentiator, not a two-point slope and not a Kalman filter.

While the ring is shorter than n, the degree drops to min(d, n_available-1).
y_m is 0 until two samples exist.

Satisfies: REQ-AIML-GIMB-002, REQ-GIMB-HIGH-001.
"""

from __future__ import annotations

# third-party
import numpy as np


def fit_rate(
    theta_rad: tuple[float, ...],
    dt_s: float,
    n_omega: int = 7,
    degree: int = 2,
) -> float:
    """Estimate angular rate at the newest encoder sample, rad/s.

    Inputs:
        theta_rad: Encoder elevations in radians, oldest to newest.
        dt_s: Uniform inner-loop period in seconds.
        n_omega: Maximum ring length used in the fit.
        degree: Polynomial degree (dropped while the ring is short).

    Outputs:
        float: y_m = d(theta)/dt at the newest sample, rad/s. 0.0 if fewer than
        two samples or dt_s is not positive.
    """
    if dt_s <= 0.0 or len(theta_rad) < 2:
        return 0.0
    window = theta_rad[-min(len(theta_rad), n_omega) :]
    n = len(window)
    deg = min(degree, n - 1)
    if deg < 1:
        return 0.0
    tau = dt_s * np.arange(1 - n, 1, dtype=np.float64)
    vandermonde = np.vander(tau, N=deg + 1, increasing=True)
    y = np.asarray(window, dtype=np.float64)
    try:
        coeffs, _, _, _ = np.linalg.lstsq(vandermonde, y, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0
    return float(coeffs[1])
