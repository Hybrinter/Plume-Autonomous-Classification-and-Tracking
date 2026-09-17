"""Wind models. Wind advects the plume column only, not the ISS or gimbal."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class WindModel(Protocol):
    """ECEF wind at a point and time."""

    def velocity_ecef_m_s(
        self, utc_s: float, pos_ecef_m: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        """Return ECEF velocity meters per second."""
        ...


@dataclass(frozen=True, slots=True)
class StillWind:
    """Zero wind."""

    def velocity_ecef_m_s(
        self, utc_s: float, pos_ecef_m: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        """Return (0, 0, 0)."""
        del utc_s, pos_ecef_m
        return (0.0, 0.0, 0.0)


@dataclass(frozen=True, slots=True)
class ConstantEcefWind:
    """Constant ECEF wind vector."""

    vx_m_s: float
    vy_m_s: float
    vz_m_s: float

    def velocity_ecef_m_s(
        self, utc_s: float, pos_ecef_m: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        """Return the configured ECEF velocity."""
        del utc_s, pos_ecef_m
        return (self.vx_m_s, self.vy_m_s, self.vz_m_s)
