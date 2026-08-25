"""Smoke test for the flight package version accessor."""

from flight.libs.version import flight_version


def test_flight_version_returns_semver() -> None:
    """flight_version returns the expected version string."""
    assert flight_version() == "0.1.0"
