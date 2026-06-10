"""Tests for the GimbalRequest pure-core command value."""

from flight.libs.types import GimbalCommandMode
from flight.payload.gimbal import GimbalRequest


def test_gimbal_request_carries_mode_and_values() -> None:
    """GimbalRequest is a frozen value: mode + two axis values + reason."""
    req = GimbalRequest(
        mode=GimbalCommandMode.RATE, az_deg=1.5, el_deg=-0.5, reason="tracking_target"
    )
    assert req.mode is GimbalCommandMode.RATE
    assert req.az_deg == 1.5
