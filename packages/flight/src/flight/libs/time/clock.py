"""Clock abstraction: time is injected, never read inside pure logic.

Separates monotonic time (control intervals, timeouts, rate limits) from wall-clock
time (message timestamps), mirroring how the existing code sources time. Pure
functions and app shells receive a Clock; the composition root owns the concrete
instance. ManualClock makes time deterministic and advanceable in tests.
"""

import time as _time
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    """Injected time source."""

    def monotonic_s(self) -> float:
        """Monotonic seconds since an arbitrary epoch (intervals, timeouts, rates)."""
        ...

    def wall_clock_iso(self) -> str:
        """Current UTC time as ISO 8601 with millisecond precision (message stamps)."""
        ...


class RealClock:
    """Production clock backed by time.monotonic() and the system UTC clock."""

    def monotonic_s(self) -> float:
        """Return time.monotonic() in seconds."""
        return _time.monotonic()

    def wall_clock_iso(self) -> str:
        """Return current UTC time as 'YYYY-MM-DDTHH:MM:SS.mmmZ'."""
        return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class ManualClock:
    """Deterministic clock for tests; monotonic time is advanced explicitly."""

    def __init__(
        self,
        monotonic_s: float = 0.0,
        wall_clock: str = "2026-01-01T00:00:00.000Z",
    ) -> None:
        """Initialize the manual clock.

        Args:
            monotonic_s: Initial monotonic seconds.
            wall_clock: Initial wall-clock ISO 8601 string.
        """
        self._monotonic_s = monotonic_s
        self._wall_clock = wall_clock

    def monotonic_s(self) -> float:
        """Return the current (manually set) monotonic seconds."""
        return self._monotonic_s

    def wall_clock_iso(self) -> str:
        """Return the current (manually set) wall-clock ISO string."""
        return self._wall_clock

    def advance(self, delta_s: float) -> None:
        """Advance monotonic time by delta_s seconds."""
        self._monotonic_s += delta_s

    def set_wall_clock(self, wall_clock: str) -> None:
        """Set the wall-clock ISO string returned by wall_clock_iso()."""
        self._wall_clock = wall_clock
