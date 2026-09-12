"""Tests for sim.trial open-loop runner and seed spawn."""

from __future__ import annotations

import math

import numpy as np
from flight.libs.config import EphemerisConfig
from sim.environment.config import EnvironmentConfig, PoissonLatitudeParams
from sim.environment.models.plume import _along_track_ahead_m, mean_encounter_time_s
from sim.trial import TrialSpec, run_open_loop, shutter_pose_at, spawn_trial_rngs
from sim.trial.spec import RewindThenLimbParams


def test_spawn_trial_rngs_are_independent() -> None:
    """Two trials from one master seed are not the same stream."""
    a, b = spawn_trial_rngs(1, 2)
    assert a.random() != b.random()


def test_run_open_loop_reproducible() -> None:
    """The same spec yields identical first-step CoGs."""
    world = EnvironmentConfig(
        plume="poisson_latitude",
        poisson_latitude=PoissonLatitudeParams(
            signed_lat_deg=(-90.0, 90.0),
            dens_per_km2=(1.0e-3, 1.0e-3),
        ),
    )
    spec = TrialSpec(n_trials=2, master_seed=11, world=world, steps=3, dt_s=1.0)
    first = run_open_loop(spec)
    second = run_open_loop(spec)
    assert len(first) == 2
    assert len(first[0].steps) == 3
    assert first[0].steps[0].truth.plume.cog_ecef_m == second[0].steps[0].truth.plume.cog_ecef_m
    assert first[1].steps[0].truth.plume.cog_ecef_m == second[1].steps[0].truth.plume.cog_ecef_m


def test_rewind_then_limb_reaches_limb() -> None:
    """Rewind shutter holds at el_limb_rad after the slew."""
    spec = TrialSpec(
        n_trials=1,
        master_seed=0,
        steps=1,
        shutter="rewind_then_limb",
        rewind=RewindThenLimbParams(
            el_start_rad=0.0,
            el_limb_rad=0.2,
            omega_img_rad_s=0.1,
        ),
    )
    early = shutter_pose_at(spec, 1.0)
    late = shutter_pose_at(spec, 10.0)
    assert abs(early.true_el_rad - 0.1) < 1e-12
    assert abs(late.true_el_rad - 0.2) < 1e-12
    assert late.true_el_rate_rad_s == 0.0


def test_mean_time_to_nadir_matches_encounter_formula() -> None:
    """MC mean first-CoG along-track wait matches 1/(lambda*v)."""
    dens = 2.0e-3
    half_km = 5.0
    world = EnvironmentConfig(
        plume="poisson_latitude",
        poisson_latitude=PoissonLatitudeParams(
            signed_lat_deg=(-90.0, 90.0),
            dens_per_km2=(dens, dens),
            cross_track_half_km=half_km,
        ),
    )
    spec = TrialSpec(
        n_trials=80,
        master_seed=42,
        world=world,
        steps=1,
        dt_s=1.0,
        shutter="constant",
        true_el_rad=0.0,
    )
    eph = EphemerisConfig()
    records = run_open_loop(spec, eph=eph)
    iss = records[0].steps[0].truth.iss
    r_m = float(np.linalg.norm(np.asarray(iss.r_m, dtype=np.float64)))
    v_m_s = float(np.linalg.norm(np.asarray(iss.v_m_s, dtype=np.float64)))
    ground_km_s = (v_m_s * (eph.wgs84_a_m / r_m)) / 1000.0
    times: list[float] = []
    for trial in records:
        first = trial.steps[0]
        cog = first.truth.plume.cog_ecef_m
        if not first.truth.plume.present or cog is None:
            continue
        ahead_m = _along_track_ahead_m(first.truth.iss, cog, eph.omega_earth_rad_s, eph.epoch_utc_s)
        if ahead_m <= 0.0:
            continue
        times.append((ahead_m / 1000.0) / ground_km_s)
    assert len(times) >= 40
    expected = mean_encounter_time_s(dens, half_km, ground_km_s)
    mean = float(np.mean(np.asarray(times, dtype=np.float64)))
    assert math.isfinite(expected)
    assert abs(mean - expected) / expected < 0.25
