"""Geometry-window sanity for the single-axis vs dual-axis study."""

from dataclasses import replace

from analysis.studies.single_axis_vs_dual_axis_gimbal.assumptions import (
    DESIGN_LAT_DEG,
    GIMBAL_BOX,
    OFFSET_PLANT_D_KM,
    OFFSET_STACK_N,
    OPTICS_SPEC,
    TLE,
)
from analysis.studies.single_axis_vs_dual_axis_gimbal.geometry import (
    OffsetRow,
    cluster_stack_offsets_km,
    latitude_table,
    offset_times,
    one_axis_window_from_environment,
    origin_window,
)
from analysis.studies.single_axis_vs_dual_axis_gimbal.look import GimbalBox, WindowMode
from analysis.studies.single_axis_vs_dual_axis_gimbal.optics import Optics, build_optics
from analysis.studies.single_axis_vs_dual_axis_gimbal.orbit import Orbit, build_orbit
from analysis.studies.single_axis_vs_dual_axis_gimbal.tracking import SampleSpan


def test_science_window_is_shorter_than_old_60deg_stop() -> None:
    """eta_max = 45 deg yields ~60 s, not the old 124 s 60-deg window."""
    optics = build_optics(OPTICS_SPEC)
    orbit = build_orbit(TLE, use_perigee=False)
    times, _ = origin_window(orbit, optics, GIMBAL_BOX, DESIGN_LAT_DEG, 0.0)
    assert 40.0 < times.along_track_s < 90.0
    assert times.gsd_band_along_start_m < 55.0
    assert GIMBAL_BOX.el_limb_deg == 45.0


def test_equator_one_axis_loses_origin_to_earth_rotation() -> None:
    """At lat 0, Earth-rotation az walk takes a point origin out of the chip."""
    optics = build_optics(OPTICS_SPEC)
    orbit = build_orbit(TLE, use_perigee=False)
    rows = latitude_table(orbit, optics, GIMBAL_BOX, (0.0,), 0.0)
    assert rows[0].one_axis_s < rows[0].two_axis_s
    assert rows[0].two_axis_s > 50.0
    assert rows[0].az_max_deg > optics.half_az_deg


def test_cluster_stack_offsets_span_plant() -> None:
    """n stacks sit on a cross-track line from -D to +D."""
    offs = cluster_stack_offsets_km(OFFSET_STACK_N, OFFSET_PLANT_D_KM)
    assert len(offs) == OFFSET_STACK_N
    assert offs[0] == -OFFSET_PLANT_D_KM
    assert offs[-1] == OFFSET_PLANT_D_KM
    assert cluster_stack_offsets_km(1, OFFSET_PLANT_D_KM) == (0.0,)


def _row_at(rows: list[OffsetRow], y_km: float) -> OffsetRow:
    """Return the offset row nearest ``y_km``."""
    return min(rows, key=lambda row: abs(row.origin_cross_km - y_km))


def test_offset_plume_seconds_innermost_survives() -> None:
    """Centroid-out still yields plume-seconds until the innermost plume exits."""
    optics = build_optics(OPTICS_SPEC)
    orbit = build_orbit(TLE, use_perigee=False)
    times, _ = origin_window(orbit, optics, GIMBAL_BOX, DESIGN_LAT_DEG, 0.0)
    rows = offset_times(orbit, optics, GIMBAL_BOX, DESIGN_LAT_DEG)
    on = _row_at(rows, 0.0)
    mid = _row_at(rows, 15.0)
    far = _row_at(rows, 40.0)
    full = OFFSET_STACK_N * times.along_track_s
    assert abs(on.one_axis_plume_s - on.two_axis_plume_s) < 2.0
    assert on.one_axis_plume_s > 0.85 * full
    assert on.one_axis_n_in == OFFSET_STACK_N
    assert 0.0 < mid.one_axis_plume_s < on.one_axis_plume_s - 1.0
    assert mid.one_axis_n_in < OFFSET_STACK_N
    assert mid.one_axis_n_in >= 1
    assert mid.two_axis_plume_s > 0.85 * full
    assert far.one_axis_plume_s < 1.0
    assert far.one_axis_n_in == 0
    assert far.two_axis_plume_s > 0.85 * full
    assert far.two_axis_n_in == OFFSET_STACK_N


def _env_matches_origin_window(
    orbit: Orbit,
    optics: Optics,
    box: GimbalBox,
    *,
    span: SampleSpan | None = None,
    dt_s: float = 1.0,
) -> tuple[float, float]:
    """Return (evaluate window, sample_pass window) and require 2*dt agreement."""
    used = span if span is not None else SampleSpan(dt_s=dt_s)
    times, _ = origin_window(orbit, optics, box, 0.0, 0.0, span=used)
    env_s = one_axis_window_from_environment(
        orbit,
        box,
        dt_s=used.dt_s,
        t_min_s=used.t_min_s,
        t_max_s=used.t_max_s,
    )
    assert abs(env_s - times.along_track_s) <= 2.0 * used.dt_s
    assert times.t_start_s > used.t_min_s
    assert times.t_stop_s < used.t_max_s
    return env_s, times.along_track_s


def test_one_axis_window_from_environment_matches_sample_pass() -> None:
    """Equator elevation window from evaluate matches the km/deg sampler."""
    optics = build_optics(OPTICS_SPEC)
    orbit = build_orbit(TLE, use_perigee=False)
    env45, _ = _env_matches_origin_window(orbit, optics, GIMBAL_BOX)
    limb60 = replace(GIMBAL_BOX, el_limb_deg=60.0)
    env60, _ = _env_matches_origin_window(orbit, optics, limb60)
    assert env60 < env45 - 1.0


def test_one_axis_window_from_environment_two_sided() -> None:
    """TWO_SIDED includes look-back on a span that does not clip t_stop."""
    optics = build_optics(OPTICS_SPEC)
    orbit = build_orbit(TLE, use_perigee=False)
    two = replace(GIMBAL_BOX, window_mode=WindowMode.TWO_SIDED)
    span = SampleSpan(t_min_s=-250.0, t_max_s=250.0, dt_s=1.0)
    env_two, _ = _env_matches_origin_window(orbit, optics, two, span=span)
    env_one, _ = _env_matches_origin_window(orbit, optics, GIMBAL_BOX, span=span)
    assert env_two > env_one + 10.0


def test_one_axis_window_from_environment_uses_supplied_orbit() -> None:
    """A perigee Orbit is flown, not the global TLE SMA."""
    optics = build_optics(OPTICS_SPEC)
    sma = build_orbit(TLE, use_perigee=False)
    peri = build_orbit(TLE, use_perigee=True)
    env_sma = one_axis_window_from_environment(sma, GIMBAL_BOX, dt_s=1.0)
    env_peri = one_axis_window_from_environment(peri, GIMBAL_BOX, dt_s=1.0)
    assert env_peri < env_sma
    _env_matches_origin_window(peri, optics, GIMBAL_BOX)
