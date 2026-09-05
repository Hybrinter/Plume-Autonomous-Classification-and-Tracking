"""Tests for the inner PI + computed-torque law."""

from flight.payload.gimbal.inner import inner_step


def test_zero_error_holds_integrator() -> None:
    """Matched r and y_m produce friction-compensation torque and frozen I at zero error."""
    result = inner_step(
        r_rad_s=0.1,
        y_m=0.1,
        integrator=0.0,
        dt_s=0.001,
        j_hat=0.008,
        b_hat=0.04,
        kp=200.0,
        ki=10_000.0,
        tau_max_nm=1.0,
        stopped=False,
    )
    assert abs(result.tau_nm - 0.04 * 0.1) < 1e-12
    assert result.integrator == 0.0
    assert result.clipped is False


def test_clip_freezes_integrator() -> None:
    """Torque saturation freezes the integrator."""
    first = inner_step(
        r_rad_s=10.0,
        y_m=0.0,
        integrator=0.0,
        dt_s=0.001,
        j_hat=0.008,
        b_hat=0.04,
        kp=200.0,
        ki=10_000.0,
        tau_max_nm=1.0,
        stopped=False,
    )
    assert first.clipped is True
    assert first.tau_nm == 1.0
    assert first.integrator == 0.0
    second = inner_step(
        r_rad_s=10.0,
        y_m=0.0,
        integrator=first.integrator,
        dt_s=0.001,
        j_hat=0.008,
        b_hat=0.04,
        kp=200.0,
        ki=10_000.0,
        tau_max_nm=1.0,
        stopped=False,
    )
    assert second.integrator == 0.0


def test_stop_freezes_integrator() -> None:
    """A hardware travel stop freezes the integrator even without torque clip."""
    result = inner_step(
        r_rad_s=0.05,
        y_m=0.0,
        integrator=0.2,
        dt_s=0.001,
        j_hat=0.008,
        b_hat=0.04,
        kp=1.0,
        ki=1.0,
        tau_max_nm=10.0,
        stopped=True,
    )
    assert result.integrator == 0.2
    assert result.clipped is False
