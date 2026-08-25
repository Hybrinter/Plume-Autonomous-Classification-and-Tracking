"""Tests for the Clock abstraction."""

import re

from flight.libs.time import Clock, ManualClock, RealClock


def test_real_clock_monotonic_non_decreasing() -> None:
    """RealClock.monotonic_s is non-decreasing across calls."""
    clock = RealClock()
    first = clock.monotonic_s()
    second = clock.monotonic_s()
    assert second >= first


def test_real_clock_wall_clock_format() -> None:
    """RealClock.wall_clock_iso returns a millisecond ISO 8601 UTC string."""
    stamp = RealClock().wall_clock_iso()
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$", stamp)


def test_manual_clock_advances() -> None:
    """ManualClock advances monotonic time only when told to."""
    clock = ManualClock(monotonic_s=10.0)
    assert clock.monotonic_s() == 10.0
    clock.advance(2.5)
    assert clock.monotonic_s() == 12.5


def test_manual_clock_wall_clock_settable() -> None:
    """ManualClock wall clock is fixed until explicitly set."""
    clock = ManualClock(wall_clock="2026-05-31T00:00:00.000Z")
    assert clock.wall_clock_iso() == "2026-05-31T00:00:00.000Z"
    clock.set_wall_clock("2026-06-01T00:00:00.000Z")
    assert clock.wall_clock_iso() == "2026-06-01T00:00:00.000Z"


def test_clocks_satisfy_protocol() -> None:
    """Both clocks conform to the Clock protocol (typed + runtime)."""
    real: Clock = RealClock()
    manual: Clock = ManualClock()
    assert isinstance(real, Clock)
    assert isinstance(manual, Clock)
