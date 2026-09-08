"""RealGimbal torque-stub tests (no serial PTU, no pyserial)."""

from flight.hal.drivers_real import RealGimbal
from flight.libs.config import GimbalConfig
from flight.libs.time import ManualClock
from flight.libs.types import Err, FaultCode, Ok


def test_set_torque_refuses_unimplemented_amp() -> None:
    """The real stub must not claim motor containment or drive authority."""
    gimbal = RealGimbal(clock=ManualClock(), cfg=GimbalConfig())
    result = gimbal.set_torque(0.5, valid_until_s=1.0)
    assert isinstance(result, Err)
    assert result.error is FaultCode.GIMBAL_FAULT


def test_goto_angle_latches_target_not_encoder() -> None:
    """goto_angle stores a travel-clamped target; encoder stays at 0."""
    clock = ManualClock()
    gimbal = RealGimbal(clock=clock, cfg=GimbalConfig())
    assert isinstance(gimbal.goto_angle(500.0), Ok)
    pos = gimbal.read_position()
    assert isinstance(pos, Ok)
    assert pos.value.el_deg == 0.0
    assert gimbal._target_el_deg == 90.0
    assert pos.value.timestamp_s == clock.monotonic_s()
    assert not hasattr(pos.value, "az_deg")


def test_stow_does_not_teleport_encoder() -> None:
    """stow() latches the stow target; encoder and switch stay at the physical pose."""
    gimbal = RealGimbal(clock=ManualClock(), cfg=GimbalConfig())
    assert isinstance(gimbal.stow(), Ok)
    pos = gimbal.read_position()
    assert isinstance(pos, Ok)
    assert pos.value.el_deg == 0.0
    switch = gimbal.read_stow_switch()
    assert isinstance(switch, Ok)
    assert switch.value is False
