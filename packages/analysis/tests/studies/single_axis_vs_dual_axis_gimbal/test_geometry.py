"""Geometry-window sanity for the single-axis vs dual-axis study."""

from analysis.lib.optics import build_optics
from analysis.lib.orbit import build_orbit
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
    origin_window,
)


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
