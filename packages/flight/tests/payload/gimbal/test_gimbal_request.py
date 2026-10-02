"""Tests for the GimbalRequest pose-command value and control references."""

import math
from dataclasses import FrozenInstanceError

import pytest
from flight.libs.types import Err, FaultCode, GimbalCommandMode, Ok
from flight.payload.gimbal import (
    GimbalRequest,
    InhibitReference,
    PoseReference,
    RateReference,
    StowReference,
    TravelEnvelope,
    validate_reference,
)


def test_gimbal_request_carries_mode_and_elevation() -> None:
    """GimbalRequest is a frozen pose value: mode + elevation + reason."""
    req = GimbalRequest(mode=GimbalCommandMode.ABSOLUTE, el_deg=-12.5, reason="goto_science")
    assert req.mode is GimbalCommandMode.ABSOLUTE
    assert req.el_deg == -12.5
    assert not hasattr(req, "az_deg")


def _envelope(
    theta_min_rad: float = 0.0,
    theta_max_rad: float = 1.0,
    omega_max_rad_s: float = 0.2,
) -> TravelEnvelope:
    """A valid default travel envelope."""
    return TravelEnvelope(
        theta_min_rad=theta_min_rad,
        theta_max_rad=theta_max_rad,
        omega_max_rad_s=omega_max_rad_s,
    )


def test_references_are_frozen_slots_values() -> None:
    """Reference and envelope records are frozen and slot-typed."""
    env = _envelope()
    rate = RateReference(rate_rad_s=0.1, envelope=env)
    with pytest.raises(FrozenInstanceError):
        rate.__setattr__("rate_rad_s", 0.2)
    with pytest.raises((AttributeError, TypeError)):
        rate.__setattr__("extra", 1)
    assert isinstance(validate_reference(rate), Ok)


def test_envelope_rejects_nan_infinite_reversed_and_zero_rate() -> None:
    """Non-finite, reversed, or zero-rate envelopes are invalid."""
    bad = [
        _envelope(theta_min_rad=math.nan),
        _envelope(theta_max_rad=math.inf),
        _envelope(omega_max_rad_s=math.nan),
        _envelope(theta_min_rad=1.0, theta_max_rad=0.0),
        _envelope(theta_min_rad=0.5, theta_max_rad=0.5),
        _envelope(omega_max_rad_s=0.0),
        _envelope(omega_max_rad_s=-0.1),
        _envelope(omega_max_rad_s=math.inf),
    ]
    for env in bad:
        result = validate_reference(RateReference(rate_rad_s=0.1, envelope=env))
        assert isinstance(result, Err)
        assert result.error is FaultCode.COMMAND_INVALID


def test_rate_reference_accepts_rate_above_cap() -> None:
    """A requested rate may exceed the envelope cap; execution clips it."""
    result = validate_reference(RateReference(rate_rad_s=5.0, envelope=_envelope()))
    assert isinstance(result, Ok)


def test_rate_reference_rejects_nonfinite_rate() -> None:
    """NaN and infinite rates are invalid."""
    for rate in (math.nan, math.inf, -math.inf):
        result = validate_reference(RateReference(rate_rad_s=rate, envelope=_envelope()))
        assert isinstance(result, Err)
        assert result.error is FaultCode.COMMAND_INVALID


def test_pose_reference_must_be_inside_envelope_inclusive() -> None:
    """Pose targets at the bounds pass; targets outside fail."""
    assert isinstance(validate_reference(PoseReference(target_rad=0.0, envelope=_envelope())), Ok)
    assert isinstance(validate_reference(PoseReference(target_rad=1.0, envelope=_envelope())), Ok)
    for target in (-0.01, 1.01, math.nan):
        result = validate_reference(PoseReference(target_rad=target, envelope=_envelope()))
        assert isinstance(result, Err)
        assert result.error is FaultCode.COMMAND_INVALID


def test_stow_reference_requires_finite_positive_timeout() -> None:
    """Stow needs an in-envelope target and a finite positive timeout."""
    env = _envelope()
    assert isinstance(
        validate_reference(StowReference(target_rad=0.5, envelope=env, timeout_s=30.0)),
        Ok,
    )
    for timeout in (0.0, -1.0, math.inf, math.nan):
        result = validate_reference(StowReference(target_rad=0.5, envelope=env, timeout_s=timeout))
        assert isinstance(result, Err)
        assert result.error is FaultCode.COMMAND_INVALID
    out_of_range = validate_reference(StowReference(target_rad=2.0, envelope=env, timeout_s=30.0))
    assert isinstance(out_of_range, Err)


def test_inhibit_reference_requires_nonempty_reason() -> None:
    """Inhibit is valid with a reason and invalid with an empty one."""
    assert isinstance(validate_reference(InhibitReference(reason="safe")), Ok)
    result = validate_reference(InhibitReference(reason=""))
    assert isinstance(result, Err)
    assert result.error is FaultCode.COMMAND_INVALID
