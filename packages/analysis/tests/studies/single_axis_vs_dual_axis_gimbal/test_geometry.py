"""Geometry-window sanity for the single-axis vs dual-axis study."""

from analysis.lib.optics import build_optics
from analysis.lib.orbit import build_orbit
from analysis.studies.single_axis_vs_dual_axis_gimbal.assumptions import (
    DESIGN_LAT_DEG,
    GIMBAL_BOX,
    OPTICS_SPEC,
    TLE,
)
from analysis.studies.single_axis_vs_dual_axis_gimbal.geometry import latitude_table, origin_window


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
