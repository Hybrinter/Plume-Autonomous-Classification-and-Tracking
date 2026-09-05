"""Tests for the real ISS ephemeris stub."""

from flight.hal.drivers_real import RealIssEphemeris
from flight.libs.types import Err, FaultCode


def test_read_state_returns_ephemeris_fault() -> None:
    """The real stub always returns Err(EPHEMERIS_FAULT)."""
    result = RealIssEphemeris().read_state(0.0)
    assert isinstance(result, Err)
    assert result.error is FaultCode.EPHEMERIS_FAULT
