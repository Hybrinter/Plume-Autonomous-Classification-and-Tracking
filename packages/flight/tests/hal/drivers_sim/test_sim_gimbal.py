"""Tests for SimGimbal first-order dynamics, limits, and the closed-loop HAL surface."""

import math

import pytest
from flight.hal.drivers_sim import SimGimbal
from flight.libs.config import GimbalConfig
from flight.libs.time import ManualClock
from flight.libs.types import Ok


def _gimbal(clock: ManualClock, **cfg_overrides: float) -> SimGimbal:
    """Construct a noiseless SimGimbal with optional GimbalConfig overrides."""
    # cfg_overrides only ever carries float fields; the ignore covers GimbalConfig's
    # heterogeneous (int/str) fields that the **splat cannot narrow.
    cfg = GimbalConfig(sim_encoder_noise_deg=0.0, **cfg_overrides)  # type: ignore[arg-type]
    return SimGimbal(clock=clock, cfg=cfg)


def test_goto_angle_approaches_target_with_lag() -> None:
    """An absolute command moves the gimbal toward the target, not instantly onto it."""
    clock = ManualClock()
    gimbal = _gimbal(clock)
    assert isinstance(gimbal.goto_angle(10.0, 0.0), Ok)
    clock.advance(0.1)
    mid = gimbal.read_position()
    assert isinstance(mid, Ok)
    assert 0.0 < mid.value.az_deg < 10.0
    clock.advance(30.0)
    settled = gimbal.read_position()
    assert isinstance(settled, Ok)
    assert abs(settled.value.az_deg - 10.0) < 0.1


def test_slew_rate_is_limited() -> None:
    """Motion toward a far target never exceeds the hardware slew envelope."""
    clock = ManualClock()
    gimbal = _gimbal(clock, max_hw_slew_rate_deg_per_s=10.0, sim_time_constant_s=0.001)
    gimbal.goto_angle(90.0, 0.0)
    clock.advance(1.0)
    pos = gimbal.read_position()
    assert isinstance(pos, Ok)
    assert pos.value.az_deg <= 10.0 + 1e-6


def test_set_rate_integrates_and_clamps_travel() -> None:
    """Rate commands integrate position and stop at the travel limit."""
    clock = ManualClock()
    gimbal = _gimbal(clock, az_max_deg=5.0)
    assert isinstance(gimbal.set_rate(2.0, 0.0), Ok)
    clock.advance(1.0)
    pos = gimbal.read_position()
    assert isinstance(pos, Ok)
    assert abs(pos.value.az_deg - 2.0) < 1e-6
    clock.advance(10.0)
    clamped = gimbal.read_position()
    assert isinstance(clamped, Ok)
    assert clamped.value.az_deg == 5.0


def test_stow_reaches_pose_and_sets_switch() -> None:
    """stow() drives to the configured stow pose; the switch reads True on arrival."""
    clock = ManualClock()
    gimbal = _gimbal(clock)
    assert isinstance(gimbal.stow(), Ok)
    early = gimbal.read_stow_switch()
    assert isinstance(early, Ok)
    assert early.value is False
    clock.advance(60.0)
    done = gimbal.read_stow_switch()
    assert isinstance(done, Ok)
    assert done.value is True
    pos = gimbal.read_position()
    assert isinstance(pos, Ok)
    assert abs(pos.value.el_deg - (-45.0)) < 0.5


def test_read_position_is_timestamped() -> None:
    """Encoder reads carry the monotonic read time."""
    clock = ManualClock()
    gimbal = _gimbal(clock)
    clock.advance(3.5)
    pos = gimbal.read_position()
    assert isinstance(pos, Ok)
    assert pos.value.timestamp_s == clock.monotonic_s()


def test_set_torque_accelerates_per_inertia() -> None:
    """A constant azimuth torque on a diagonal J with no damping raises az rate and pose."""
    clock = ManualClock()
    cfg = GimbalConfig(
        sim_encoder_noise_deg=0.0,
        J_kg_m2=(1.0, 0.0, 0.0, 1.0),
        B_nms_per_rad=(0.0, 0.0, 0.0, 0.0),
    )
    gimbal = SimGimbal(clock=clock, cfg=cfg)
    assert isinstance(gimbal.set_torque(1.0, 0.0), Ok)
    dt = 0.1
    clock.advance(dt)
    pos = gimbal.read_position()
    assert isinstance(pos, Ok)
    omega_az = 1.0 * dt  # rad/s from ω += J^{-1} τ dt
    expected_az_deg = omega_az * dt * (180.0 / math.pi)
    assert pos.value.az_deg == pytest.approx(expected_az_deg, rel=1e-9)
    assert pos.value.el_deg == pytest.approx(0.0, abs=1e-12)


def test_set_torque_couples_axes_when_j_has_off_diagonal() -> None:
    """Off-diagonal inertia produces elevation motion from an azimuth-only torque."""
    clock = ManualClock()
    cfg = GimbalConfig(
        sim_encoder_noise_deg=0.0,
        J_kg_m2=(1.0, 0.2, 0.2, 1.0),
        B_nms_per_rad=(0.0, 0.0, 0.0, 0.0),
    )
    gimbal = SimGimbal(clock=clock, cfg=cfg)
    gimbal.set_torque(1.0, 0.0)
    clock.advance(0.1)
    pos = gimbal.read_position()
    assert isinstance(pos, Ok)
    assert pos.value.az_deg > 0.0
    assert pos.value.el_deg < 0.0


def test_set_torque_travel_clamp_zeros_rate() -> None:
    """Torque into a travel stop clamps pose and holds it on further time advance."""
    clock = ManualClock()
    cfg = GimbalConfig(
        sim_encoder_noise_deg=0.0,
        az_max_deg=1.0,
        J_kg_m2=(1.0, 0.0, 0.0, 1.0),
        B_nms_per_rad=(0.0, 0.0, 0.0, 0.0),
    )
    gimbal = SimGimbal(clock=clock, cfg=cfg)
    gimbal.set_torque(20.0, 0.0)
    clock.advance(2.0)
    hit = gimbal.read_position()
    assert isinstance(hit, Ok)
    assert hit.value.az_deg == 1.0
    clock.advance(2.0)
    held = gimbal.read_position()
    assert isinstance(held, Ok)
    assert held.value.az_deg == 1.0
