"""RadiusProfile Poisson tables and sim encounter-time helper."""

import math

import numpy as np
from analysis.studies.single_axis_vs_dual_axis_gimbal.profile import (
    RadiusProfile,
    encounter_time_s,
)
from sim.environment.models.plume import mean_encounter_time_s


def test_encounter_time_uses_sim_helper() -> None:
    """Swath width maps to corridor half-width for mean_encounter_time_s."""
    dens = 2.0e-3
    swath = 10.0
    speed = 7.4
    got = encounter_time_s(dens, swath, speed)
    expected = mean_encounter_time_s(dens, 0.5 * swath, speed)
    assert got == expected
    assert math.isfinite(got)


def test_poisson_latitude_params_from_profile() -> None:
    """Signed-latitude density fills PoissonLatitudeParams."""
    profile = RadiusProfile(
        abs_lat=np.array([1.0]),
        mean_r_km=np.array([3.0]),
        mean_d_km=np.array([1.0]),
        rows=(),
        signed_lat=np.array([-20.0, 20.0]),
        stack_dens_per_km2=np.array([1.0e-4, 3.0e-4]),
    )
    params = profile.poisson_latitude_params(cross_track_half_km=4.0)
    assert params.signed_lat_deg == (-20.0, 20.0)
    assert params.dens_per_km2 == (1.0e-4, 3.0e-4)
    assert params.cross_track_half_km == 4.0
