"""Tests for the GimbalRequest pose-command value."""

from flight.libs.types import GimbalCommandMode
from flight.payload.gimbal import GimbalRequest


def test_gimbal_request_carries_mode_and_elevation() -> None:
    """GimbalRequest is a frozen pose value: mode + elevation + reason."""
    req = GimbalRequest(mode=GimbalCommandMode.ABSOLUTE, el_deg=-12.5, reason="goto_science")
    assert req.mode is GimbalCommandMode.ABSOLUTE
    assert req.el_deg == -12.5
    assert not hasattr(req, "az_deg")
