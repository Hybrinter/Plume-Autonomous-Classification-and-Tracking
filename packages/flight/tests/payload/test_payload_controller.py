"""Tests for the mode-free ServoController: reference mapping and inner PI."""

import math

import pytest
from flight.libs.config import ControllerConfig, GimbalConfig
from flight.libs.types import Ok
from flight.payload.control import ServoController, ServoState
from flight.payload.gimbal.request import (
    InhibitReference,
    PoseReference,
    RateReference,
    StowReference,
    TravelEnvelope,
)
from flight.payload.tracking import EncoderSample

_CFG = ControllerConfig()
_GIMBAL = GimbalConfig()
_HW = TravelEnvelope(
    theta_min_rad=math.radians(_GIMBAL.el_hw_min_deg),
    theta_max_rad=math.radians(_GIMBAL.el_hw_max_deg),
    omega_max_rad_s=math.radians(_GIMBAL.max_hw_slew_rate_deg_per_s),
)


def _servo() -> ServoController:
    """Build a servo controller with default controller + gimbal geometry."""
    return ServoController.from_config(_CFG, _GIMBAL)


def _encoder(t_s: float, angle_rad: float = 0.0) -> EncoderSample:
    """Build a timestamped encoder sample for one inner tick."""
    return EncoderSample(f"encoder:{t_s:.6f}", t_s, angle_rad, 0.0)


def test_initial_state_is_cold() -> None:
    """The servo boots with an empty ring, zero integrator, zero commanded rate."""
    state = _servo().initial_state()
    assert state.encoder.samples == ()
    assert state.encoder.last_theta_enc_rad is None
    assert state.inner.integrator == 0.0
    assert state.integrity.freeze_strikes == 0
    assert state.commanded_rate_rad_s == 0.0


def test_reference_rate_inhibit_is_zero() -> None:
    """InhibitReference maps to zero rate regardless of attitude."""
    servo = _servo()
    rate = servo.reference_rate(InhibitReference("safe"), theta_rad=0.0)
    assert isinstance(rate, Ok)
    assert rate.value == 0.0


def test_reference_rate_invalid_reference_errors() -> None:
    """A reference failing validation returns Err rather than a rate."""
    servo = _servo()
    bad = InhibitReference("")
    rate = servo.reference_rate(bad, theta_rad=0.0)
    assert not isinstance(rate, Ok)


def test_reference_rate_pose_uses_position_law() -> None:
    """PoseReference commands a positive rate toward an above-boresight target."""
    servo = _servo()
    target = math.radians(10.0)
    pose = PoseReference(target_rad=target, envelope=_HW)
    rate = servo.reference_rate(pose, theta_rad=0.0)
    assert isinstance(rate, Ok)
    assert rate.value > 0.0
    assert rate.value <= _HW.omega_max_rad_s + 1e-12


def test_reference_rate_pose_stops_at_target() -> None:
    """A PoseReference at the target commands zero rate."""
    servo = _servo()
    target = math.radians(10.0)
    pose = PoseReference(target_rad=target, envelope=_HW)
    rate = servo.reference_rate(pose, theta_rad=target)
    assert isinstance(rate, Ok)
    assert abs(rate.value) < 1e-9


def test_reference_rate_rate_reference_clips_to_envelope() -> None:
    """RateReference magnitudes are capped by the travel envelope."""
    servo = _servo()
    fast = RateReference(rate_rad_s=_HW.omega_max_rad_s * 10.0, envelope=_HW)
    rate = servo.reference_rate(fast, theta_rad=0.0)
    assert isinstance(rate, Ok)
    assert abs(rate.value) <= _HW.omega_max_rad_s + 1e-12


def test_reference_rate_stow_bounded() -> None:
    """StowReference uses the position law inside the stow envelope."""
    servo = _servo()
    env = TravelEnvelope(
        theta_min_rad=_HW.theta_min_rad,
        theta_max_rad=_HW.theta_max_rad,
        omega_max_rad_s=math.radians(_GIMBAL.xeryon.stow_reference_rate_deg_per_s),
    )
    stow = StowReference(target_rad=math.radians(_GIMBAL.stow_el_deg), envelope=env, timeout_s=30.0)
    rate = servo.reference_rate(stow, theta_rad=0.0)
    assert isinstance(rate, Ok)
    assert abs(rate.value) <= env.omega_max_rad_s + 1e-12


def test_inner_step_writes_torque() -> None:
    """inner_step with a nonzero reference rate produces a torque command."""
    servo = _servo()
    state = servo.initial_state()
    reference = RateReference(rate_rad_s=math.radians(1.0), envelope=_HW)
    tick = servo.inner_step(state, 0.001, _encoder(0.001), reference)
    assert tick.tau_nm != 0.0
    assert tick.state.commanded_rate_rad_s > 0.0
    assert len(tick.state.encoder.samples) == 1
    assert tick.state.inner.last_inner_s == 0.001


def test_lower_stop_ignores_phantom_inbound_rate() -> None:
    """A rising encoder count on the nadir stop still drives off the stop."""
    servo = _servo()
    reference = RateReference(rate_rad_s=math.radians(5.0), envelope=_HW)
    state = servo.initial_state()
    theta = math.radians(_GIMBAL.el_hw_min_deg)
    tick = None
    for i in range(1, 8):
        tick = servo.inner_step(
            state, i * 0.001, _encoder(i * 0.001, theta + 1.0e-5 * i), reference
        )
        state = tick.state
    assert tick is not None
    assert tick.tau_nm > 0.0


def test_inner_step_inhibit_resets_integrator() -> None:
    """An InhibitReference zeroes the rate and clears the dynamic PI integrator."""
    servo = _servo()
    reference = RateReference(rate_rad_s=math.radians(1.0), envelope=_HW)
    state = servo.initial_state()
    for i in range(1, 6):
        tick = servo.inner_step(state, i * 0.001, _encoder(i * 0.001), reference)
        state = tick.state
    assert state.inner.integrator != 0.0 or state.commanded_rate_rad_s != 0.0
    tick = servo.inner_step(state, 0.006, _encoder(0.006), InhibitReference("safe"))
    assert tick.state.commanded_rate_rad_s == 0.0
    assert tick.state.inner.integrator == 0.0


def test_inner_step_owns_no_graph_state() -> None:
    """ServoState carries only encoder, inner PI, integrity, and commanded rate."""
    assert set(ServoState.__dataclass_fields__) == {
        "encoder",
        "inner",
        "integrity",
        "commanded_rate_rad_s",
    }


def _guarded_theta_min() -> float:
    """Lowest theta whose outward margin to the science guard is exhausted."""
    guard = math.radians(_CFG.integrity.science_boundary_guard_deg)
    return _HW.theta_min_rad + guard


def _guarded_theta_max() -> float:
    """Highest theta whose outward margin to the science guard is exhausted."""
    guard = math.radians(_CFG.integrity.science_boundary_guard_deg)
    return _HW.theta_max_rad - guard


@pytest.mark.parametrize("detailed_plant", [True, False], ids=["finite", "production"])
def test_reference_rate_bound_outward_is_finite_zero(detailed_plant: bool) -> None:
    """At an exhausted guarded bound the outward rate is a finite zero."""
    servo = _servo()
    outward_lo = RateReference(rate_rad_s=-math.radians(5.0), envelope=_HW)
    rate = servo.reference_rate(
        outward_lo, theta_rad=_guarded_theta_min(), detailed_plant=detailed_plant
    )
    assert isinstance(rate, Ok)
    assert rate.value == 0.0
    assert math.isfinite(rate.value)
    outward_hi = RateReference(rate_rad_s=math.radians(5.0), envelope=_HW)
    rate = servo.reference_rate(
        outward_hi, theta_rad=_guarded_theta_max(), detailed_plant=detailed_plant
    )
    assert isinstance(rate, Ok)
    assert rate.value == 0.0
    assert math.isfinite(rate.value)


@pytest.mark.parametrize("detailed_plant", [True, False], ids=["finite", "production"])
def test_reference_rate_bound_inward_valid_sign(detailed_plant: bool) -> None:
    """At a guarded bound the inward rate keeps its sign and stays finite."""
    servo = _servo()
    inward = RateReference(rate_rad_s=math.radians(1.0), envelope=_HW)
    rate = servo.reference_rate(
        inward, theta_rad=_guarded_theta_min(), detailed_plant=detailed_plant
    )
    assert isinstance(rate, Ok)
    assert 0.0 < rate.value <= _HW.omega_max_rad_s + 1e-12
    inward_down = RateReference(rate_rad_s=-math.radians(1.0), envelope=_HW)
    rate = servo.reference_rate(
        inward_down, theta_rad=_guarded_theta_max(), detailed_plant=detailed_plant
    )
    assert isinstance(rate, Ok)
    assert -_HW.omega_max_rad_s - 1e-12 <= rate.value < 0.0


@pytest.mark.parametrize("detailed_plant", [True, False], ids=["finite", "production"])
def test_reference_rate_interior_capped_finite(detailed_plant: bool) -> None:
    """Interior rates remain finite and envelope-capped under both plant laws."""
    servo = _servo()
    theta = 0.5 * (_HW.theta_min_rad + _HW.theta_max_rad)
    fast = RateReference(rate_rad_s=_HW.omega_max_rad_s * 10.0, envelope=_HW)
    rate = servo.reference_rate(fast, theta_rad=theta, detailed_plant=detailed_plant)
    assert isinstance(rate, Ok)
    assert math.isfinite(rate.value)
    assert 0.0 < rate.value <= _HW.omega_max_rad_s + 1e-12


@pytest.mark.parametrize("theta", [math.nan, math.inf, -math.inf])
def test_reference_rate_nonfinite_theta_errors(theta: float) -> None:
    """A nonfinite encoder theta is COMMAND_INVALID, never a motion rate."""
    servo = _servo()
    reference = RateReference(rate_rad_s=math.radians(1.0), envelope=_HW)
    rate = servo.reference_rate(reference, theta_rad=theta)
    assert not isinstance(rate, Ok)


def test_inner_step_invalid_reference_emits_no_torque() -> None:
    """An invalid reference produces a zero tick: no braking through the PI."""
    servo = _servo()
    state = servo.initial_state()
    reference = RateReference(rate_rad_s=math.radians(1.0), envelope=_HW)
    for i in range(1, 4):
        tick = servo.inner_step(state, i * 0.001, _encoder(i * 0.001), reference)
        state = tick.state
    assert state.commanded_rate_rad_s != 0.0
    tick = servo.inner_step(state, 0.004, _encoder(0.004, math.nan), reference)
    assert tick.tau_nm == 0.0
    assert tick.state.commanded_rate_rad_s == 0.0
    assert tick.state.inner.last_tau_nm == 0.0
