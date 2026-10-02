"""Tests for GraphParameters envelopes, policy projection, and freshness."""

import math

from flight.libs.config import PactConfig
from flight.payload.graphs.parameters import GraphParameters, encoder_fresh
from flight.payload.records import ActivationKey

from .support import TickBuilder


def test_envelopes_derive_from_config(params: GraphParameters) -> None:
    """Science/pose/stow envelopes bound the configured windows and caps."""
    gimbal = params.config.gimbal
    assert params.science_envelope.theta_min_rad == math.radians(gimbal.el_science_min_deg)
    assert params.science_envelope.theta_max_rad == math.radians(gimbal.el_science_max_deg)
    assert params.pose_envelope.omega_max_rad_s == min(
        math.radians(params.config.controller.position.r_max_deg_per_s),
        math.radians(gimbal.max_hw_slew_rate_deg_per_s),
    )
    assert params.stow_envelope.omega_max_rad_s == math.radians(
        gimbal.xeryon.stow_reference_rate_deg_per_s
    )


def test_policy_limits_match_capture_config(params: GraphParameters) -> None:
    """PolicyLimits mirrors the capture config slice."""
    capture = params.config.sensor.capture
    limits = params.policy_limits
    assert limits.exposure_min_us == capture.exposure_min_us
    assert limits.exposure_max_us == capture.exposure_max_us
    assert limits.max_frame_rate_hz == capture.max_frame_rate_hz


def test_default_policy_enabled_disables_cleanly(
    params: GraphParameters,
) -> None:
    """Enabled policy acquires and infers; disabled policy does neither."""
    enabled = params.default_policy(enabled=True)
    assert enabled.imaging.acquisition_enabled is True
    assert enabled.inference.enabled is (params.config.sensor.capture.duty_cycle > 0.0)
    disabled = params.default_policy(enabled=False)
    assert disabled.imaging.acquisition_enabled is False
    assert disabled.inference.enabled is False
    assert disabled.imaging.publish_products is False


def test_detailed_plant_selects_bounded_or_unbounded_law() -> None:
    """detailed_plant=True uses tau/J and inner kp; False gives unbounded."""
    detailed = GraphParameters(config=PactConfig(), detailed_plant=True)
    assert math.isfinite(detailed.max_decel_rad_s2)
    assert detailed.rate_loop_bandwidth_rad_s == detailed.config.controller.inner.kp
    production = GraphParameters(config=PactConfig(), detailed_plant=False)
    assert production.max_decel_rad_s2 == math.inf
    assert production.rate_loop_bandwidth_rad_s == math.inf


def test_encoder_fresh_matrix(
    params: GraphParameters, tick: TickBuilder, key: ActivationKey
) -> None:
    """Missing, nonfinite, future, stale, and invalid feedback are not fresh."""
    fresh = tick(1.0, key)
    assert encoder_fresh(fresh, params) is True
    stale = tick(1.0, key, encoder_t_s=0.0)
    assert encoder_fresh(stale, params) is False
    missing = tick(1.0, key, encoder_angle_rad=None)
    assert encoder_fresh(missing, params) is False
    future = tick(1.0, key, encoder_t_s=2.0)
    assert encoder_fresh(future, params) is False


def test_encoder_fresh_rejects_bad_variance_and_out_of_range_angle(
    params: GraphParameters, tick: TickBuilder, key: ActivationKey
) -> None:
    """NaN/negative variance and out-of-hardware angles are not fresh."""
    nan_var = tick(1.0, key, encoder_variance_rad2=math.nan)
    assert encoder_fresh(nan_var, params) is False
    neg_var = tick(1.0, key, encoder_variance_rad2=-1e-9)
    assert encoder_fresh(neg_var, params) is False
    below = tick(1.0, key, encoder_angle_rad=math.radians(params.config.gimbal.el_hw_min_deg - 1.0))
    assert encoder_fresh(below, params) is False
    above = tick(1.0, key, encoder_angle_rad=math.radians(params.config.gimbal.el_hw_max_deg + 1.0))
    assert encoder_fresh(above, params) is False
    for angle_deg in (
        params.config.gimbal.el_hw_min_deg,
        params.config.gimbal.el_hw_max_deg,
    ):
        at_bound = tick(1.0, key, encoder_angle_rad=math.radians(angle_deg))
        assert encoder_fresh(at_bound, params) is True
