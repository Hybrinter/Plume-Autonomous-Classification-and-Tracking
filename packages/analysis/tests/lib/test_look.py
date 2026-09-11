"""Facade checks: analysis Look is the km/deg view of sim LookAngles."""

from __future__ import annotations

import math

import numpy as np
from analysis.lib.look import look_at, look_from_si
from analysis.lib.orbit import IssTle, argument_of_latitude, build_orbit, iss_eci, origin_ecef
from flight.hal.interfaces.ephemeris import IssState
from sim.environment.look import look_angles_at
from sim.environment.models.earth import SphereEarth
from sim.environment.records import LookAngles


def test_look_from_si_nadir_convention() -> None:
    """0 rad flight elevation maps to 90 deg analysis elevation."""
    si = LookAngles(0.0, 0.0, 0.0, 400_000.0, 0.0, True)
    look = look_from_si(si, 90.0)
    assert look.el_deg == 90.0
    assert look.slant_km == 400.0


def test_look_at_matches_sim_si() -> None:
    """look_at equals look_angles_at after the km/deg conversion."""
    tle = IssTle(
        epoch="2026-09-01",
        inclination_deg=51.6312,
        eccentricity=0.0005055,
        mean_motion_rev_per_day=15.48958602,
    )
    orbit = build_orbit(tle, use_perigee=False)
    u0 = argument_of_latitude(51.6312, orbit.inclination_rad)
    r_iss, vel = iss_eci(0.0, orbit, u0)
    tgt = origin_ecef(orbit, 51.6312)
    look = look_at(r_iss, vel, tgt, 90.0)
    iss = IssState(
        r_m=(float(r_iss[0]) * 1000.0, float(r_iss[1]) * 1000.0, float(r_iss[2]) * 1000.0),
        v_m_s=(float(vel[0]) * 1000.0, float(vel[1]) * 1000.0, float(vel[2]) * 1000.0),
        epoch_utc_s=0.0,
    )
    cog = (float(tgt[0]) * 1000.0, float(tgt[1]) * 1000.0, float(tgt[2]) * 1000.0)
    si = look_angles_at(iss, cog, SphereEarth(a_m=float(np.linalg.norm(tgt)) * 1000.0), 0.0, 0.0)
    converted = look_from_si(si, 90.0)
    assert look.visible is converted.visible
    assert abs(look.el_deg - converted.el_deg) < 1e-9
    assert abs(look.az_deg - converted.az_deg) < 1e-9
    assert abs(look.slant_km - converted.slant_km) < 1e-9
    assert math.isclose(look.el_deg, 90.0, abs_tol=1e-6)
