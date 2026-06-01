"""Injectable clock abstraction (monotonic + wall-clock)."""

from flight.libs.time.clock import Clock, ManualClock, RealClock

__all__ = ["Clock", "ManualClock", "RealClock"]
