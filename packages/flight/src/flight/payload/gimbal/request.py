"""GimbalRequest: the pure controller's typed command output.

A GimbalRequest is NOT a bus message: it flows by return value from the pure control
core to the payload app shell, which maps it onto GimbalActuator HAL calls and
publishes a GimbalCommandMsg telemetry record. Keeping the pure core ignorant of the
HAL preserves the pure-core contract.

Satisfies: REQ-AIML-GIMB-001, REQ-GIMB-HIGH-001.
"""

from __future__ import annotations

# stdlib
from dataclasses import dataclass

# internal
from flight.libs.types import GimbalCommandMode


@dataclass(frozen=True, slots=True)
class GimbalRequest:
    """One gimbal command decided by the pure control core.

    Attributes:
        mode: Interpretation of the axis values (RATE deg/s, ABSOLUTE deg, STOW/HOME
            ignore them).
        az_deg: Azimuth rate (RATE) or target azimuth (ABSOLUTE); 0.0 for STOW/HOME.
        el_deg: Elevation rate (RATE) or target elevation (ABSOLUTE); 0.0 for STOW/HOME.
        reason: Human-readable reason code for telemetry/logging.

    Notes:
        GimbalRequest never travels on the bus; it is returned by value from the pure
        control core (PayloadController.step / GimbalArbiter.step) to the PayloadApp
        shell, which maps it onto GimbalActuator HAL calls. This separation ensures the
        pure-core contract: no I/O, no bus access, no clock reads inside decision cores.
    """

    mode: GimbalCommandMode
    az_deg: float
    el_deg: float
    reason: str
