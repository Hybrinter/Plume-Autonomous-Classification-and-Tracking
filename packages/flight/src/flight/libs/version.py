"""Flight package version accessor.

Provides the flight software version string. This is the first real flight
module and exists to exercise the quality gates against typed flight code.
"""

FLIGHT_VERSION: str = "0.1.0"


def flight_version() -> str:
    """Return the flight software version string.

    Returns:
        str: The semantic version of the flight package.
    """
    return FLIGHT_VERSION
