"""Named-model EnvironmentConfig and builders.

This config is sim-only. It is not a member of PactConfig.
"""

from __future__ import annotations

# stdlib
from dataclasses import field
from typing import Literal

# internal
from flight.libs.config import EphemerisConfig
from flight.libs.types import Err, Ok, Result

# third-party
from pydantic import ConfigDict, TypeAdapter
from pydantic.dataclasses import dataclass as pydantic_dataclass

from sim.environment.evaluate import EnvironmentModels
from sim.environment.models.appearance import OracleMask
from sim.environment.models.earth import EarthModel, SphereEarth, Wgs84Ellipsoid
from sim.environment.models.optics import PinholeOptics
from sim.environment.models.orbit import CircularKepler
from sim.environment.models.plume import (
    BandplaneGaussian,
    EcefColumn,
    PlumeModel,
    PoissonLatitude,
)
from sim.environment.models.wind import ConstantEcefWind, StillWind, WindModel

_SCHEMA = ConfigDict(extra="forbid", frozen=True)

EarthName = Literal["wgs84_ellipsoid", "sphere"]
OrbitName = Literal["circular_kepler"]
WindName = Literal["still", "constant_ecef"]
PlumeName = Literal["bandplane_gaussian", "ecef_column", "poisson_latitude"]
OpticsName = Literal["pinhole"]
AppearanceName = Literal["oracle_mask"]


@pydantic_dataclass(config=_SCHEMA)
class ConstantEcefWindParams:
    """ECEF wind components in meters per second."""

    vx_m_s: float = 0.0
    vy_m_s: float = 0.0
    vz_m_s: float = 0.0


@pydantic_dataclass(config=_SCHEMA)
class EcefColumnParams:
    """Frozen ECEF CoG; None places nadir on the height proxy."""

    cog_ecef_m: tuple[float, float, float] | None = None
    along_sigma_m: float = 200.0
    cross_sigma_m: float = 200.0
    height_proxy_m: float = 2000.0


@pydantic_dataclass(config=_SCHEMA)
class PoissonLatitudeParams:
    """Signed-latitude stack density and corridor width for PoissonLatitude."""

    signed_lat_deg: tuple[float, ...] = (-90.0, 90.0)
    dens_per_km2: tuple[float, ...] = (0.0, 0.0)
    along_sigma_m: float = 200.0
    cross_sigma_m: float = 200.0
    height_proxy_m: float = 2000.0
    cross_track_half_km: float = 5.0


@pydantic_dataclass(config=_SCHEMA)
class EnvironmentConfig:
    """Named world models. Fidelity is inferred from the names, not a rung."""

    earth: EarthName = "wgs84_ellipsoid"
    orbit: OrbitName = "circular_kepler"
    wind: WindName = "still"
    plume: PlumeName = "bandplane_gaussian"
    optics: OpticsName = "pinhole"
    appearance: AppearanceName = "oracle_mask"
    constant_ecef: ConstantEcefWindParams = field(default_factory=ConstantEcefWindParams)
    ecef_column: EcefColumnParams = field(default_factory=EcefColumnParams)
    poisson_latitude: PoissonLatitudeParams = field(default_factory=PoissonLatitudeParams)


_CONFIG_ADAPTER = TypeAdapter(EnvironmentConfig)


def parse_environment_config(data: dict[str, object]) -> Result[EnvironmentConfig, str]:
    """Validate a dict into EnvironmentConfig."""
    try:
        return Ok(_CONFIG_ADAPTER.validate_python(data))
    except Exception as exc:  # pydantic ValidationError
        return Err(str(exc))


def build_models(config: EnvironmentConfig, eph: EphemerisConfig) -> Result[EnvironmentModels, str]:
    """Resolve named models. Unknown combinations cannot happen (Literals)."""
    if config.earth == "sphere":
        b_m = eph.wgs84_a_m * (1.0 - eph.wgs84_f)
        earth: EarthModel = SphereEarth(a_m=0.5 * (eph.wgs84_a_m + b_m), f=0.0)
    else:
        earth = Wgs84Ellipsoid(a_m=eph.wgs84_a_m, f=eph.wgs84_f)

    orbit = CircularKepler.from_ephemeris_config(eph)

    if config.wind == "constant_ecef":
        wind: WindModel = ConstantEcefWind(
            vx_m_s=config.constant_ecef.vx_m_s,
            vy_m_s=config.constant_ecef.vy_m_s,
            vz_m_s=config.constant_ecef.vz_m_s,
        )
    else:
        wind = StillWind()

    if config.plume == "ecef_column":
        p = config.ecef_column
        plume: PlumeModel = EcefColumn(
            cog_ecef_m=p.cog_ecef_m,
            along_sigma_m=p.along_sigma_m,
            cross_sigma_m=p.cross_sigma_m,
            height_proxy_m=p.height_proxy_m,
        )
    elif config.plume == "poisson_latitude":
        src = config.poisson_latitude
        if len(src.signed_lat_deg) != len(src.dens_per_km2) or len(src.signed_lat_deg) < 1:
            return Err("poisson_latitude tables must be non-empty and equal length")
        lats = src.signed_lat_deg
        if any(lats[i] >= lats[i + 1] for i in range(len(lats) - 1)):
            return Err("poisson_latitude signed_lat_deg must be strictly increasing")
        plume = PoissonLatitude(
            signed_lat_deg=src.signed_lat_deg,
            dens_per_km2=src.dens_per_km2,
            along_sigma_m=src.along_sigma_m,
            cross_sigma_m=src.cross_sigma_m,
            height_proxy_m=src.height_proxy_m,
            cross_track_half_km=src.cross_track_half_km,
        )
    else:
        plume = BandplaneGaussian()

    return Ok(
        EnvironmentModels(
            earth=earth,
            orbit=orbit,
            wind=wind,
            plume=plume,
            optics=PinholeOptics(),
            appearance=OracleMask(),
        )
    )
