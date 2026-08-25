"""Startup health-gate decision tests."""

from flight.core.health import missing_heartbeats, startup_healthy

_MONITORED = ("payload", "fault", "thermal")


def test_all_seen_is_healthy() -> None:
    """When every monitored subsystem has heartbeat, startup is healthy."""
    assert startup_healthy({"payload", "fault", "thermal"}, _MONITORED)
    assert missing_heartbeats({"payload", "fault", "thermal"}, _MONITORED) == set()


def test_missing_subsystem_is_unhealthy() -> None:
    """A silent monitored subsystem makes startup unhealthy and is reported missing."""
    assert not startup_healthy({"payload", "fault"}, _MONITORED)
    assert missing_heartbeats({"payload", "fault"}, _MONITORED) == {"thermal"}


def test_extra_seen_subsystems_ignored() -> None:
    """Heartbeats from unmonitored names do not affect the gate."""
    assert startup_healthy({"payload", "fault", "thermal", "extra"}, _MONITORED)
