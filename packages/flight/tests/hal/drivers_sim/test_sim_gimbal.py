"""Tests for SimGimbal rigid-body ODE plant, torque clips, and encoder reads."""

import math

from flight.hal.drivers_sim import SimGimbal
from flight.libs.config import GimbalConfig, GimbalSimulationConfig
from flight.libs.time import ManualClock
from flight.libs.types import Ok


def _gimbal(clock: ManualClock, **cfg_overrides: float) -> SimGimbal:
    """Construct a noiseless SimGimbal with optional GimbalConfig overrides."""
    tau_max = cfg_overrides.pop("tau_max_nm", 1.0)
    cfg = GimbalConfig(
        max_hw_slew_rate_deg_per_s=cfg_overrides.pop("max_hw_slew_rate_deg_per_s", 10.0),
        simulation=GimbalSimulationConfig(tau_max_nm=tau_max, encoder_noise_deg=0.0),
    )
    assert not cfg_overrides
    return SimGimbal(clock=clock, cfg=cfg, inner_dt_s=0.001)


def test_constant_torque_moves_elevation() -> None:
    """Held torque integrates the plant; elevation leaves the origin."""
    clock = ManualClock()
    gimbal = _gimbal(clock)
    assert isinstance(gimbal.set_torque(0.2, valid_until_s=2.0), Ok)
    clock.advance(1.0)
    pos = gimbal.read_position()
    assert isinstance(pos, Ok)
    assert pos.value.el_deg > 0.1
    assert not hasattr(pos.value, "az_deg")


def test_torque_clip_and_slew_cap() -> None:
    """Torque above tau_max and resulting rate stay inside the hardware envelope."""
    clock = ManualClock()
    gimbal = _gimbal(clock, tau_max_nm=1.0, max_hw_slew_rate_deg_per_s=10.0)
    gimbal.set_torque(50.0, valid_until_s=2.0)
    clock.advance(1.0)
    pos = gimbal.read_position()
    assert isinstance(pos, Ok)
    assert pos.value.el_deg <= 10.0 + 0.5
    assert abs(gimbal._tau_nm) <= 1.0 + 1e-12


def test_travel_stop_clamps_elevation() -> None:
    """Sustained torque stops at the hardware travel limit."""
    clock = ManualClock()
    gimbal = _gimbal(clock)
    gimbal.set_torque(1.0, valid_until_s=31.0)
    clock.advance(30.0)
    pos = gimbal.read_position()
    assert isinstance(pos, Ok)
    assert abs(pos.value.el_deg - 90.0) < 0.5


def test_stow_switch_requires_command_and_pose() -> None:
    """stow() arms the switch; the switch reads True only near the stow pose."""
    clock = ManualClock()
    gimbal = _gimbal(clock)
    assert isinstance(gimbal.stow(), Ok)
    early = gimbal.read_stow_switch()
    assert isinstance(early, Ok)
    assert early.value is False
    gimbal.set_torque(-1.0, valid_until_s=21.0)
    clock.advance(20.0)
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


def test_command_authority_expiry_inhibits_drive() -> None:
    """A stale host command cannot leave simulated torque energized."""
    clock = ManualClock()
    gimbal = _gimbal(clock)
    assert isinstance(gimbal.set_torque(0.2, valid_until_s=0.01), Ok)
    clock.advance(0.02)
    health = gimbal.read_health()
    assert isinstance(health, Ok)
    assert health.value.inhibit_confirmed is True
    assert gimbal._tau_nm == 0.0


def test_frozen_catch_up_does_not_double_count() -> None:
    """set_torque at a frozen clock plus a later advance of the same dt moves once."""
    clock = ManualClock()
    gimbal = _gimbal(clock)
    n = 1000
    for _ in range(n):
        gimbal.set_torque(0.1)
    after_frozen = gimbal.true_el_deg
    clock.advance(n * 0.001)
    after_advance = gimbal.read_position()
    assert isinstance(after_advance, Ok)
    assert abs(after_advance.value.el_deg - after_frozen) < 0.05
    assert math.isfinite(gimbal.true_omega_rad_s)
