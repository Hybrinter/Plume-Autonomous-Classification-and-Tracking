"""ISS ephemeris hardware abstraction.

`IssEphemeris.read_state` returns inertial ISS position and velocity in ECI meters.
The sim driver is a circular Keplerian orbit. The real driver is a stub until the
station TLE API exists.

Satisfies: REQ-AIML-GIMB-002, REQ-GIMB-HIGH-001.
"""

from __future__ import annotations

# stdlib
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

# internal
from flight.libs.types import FaultCode, Result


@dataclass(frozen=True, slots=True)
class IssState:
    """ISS inertial state in ECI, SI meters.

    Attributes:
        r_m: Position in ECI meters.
        v_m_s: Inertial velocity in ECI meters per second.
        epoch_utc_s: UTC seconds of this sample.
        frame: Frame tag; always ECI at this Protocol.
    """

    r_m: tuple[float, float, float]
    v_m_s: tuple[float, float, float]
    epoch_utc_s: float
    frame: Literal["ECI"] = "ECI"


@runtime_checkable
class IssEphemeris(Protocol):
    """Injected source of ISS ECI state for the outer predictor."""

    def read_state(self, now_utc_s: float) -> Result[IssState, FaultCode]:
        """Return ISS ECI state at the given UTC epoch, or Err on a dead source."""
        ...
