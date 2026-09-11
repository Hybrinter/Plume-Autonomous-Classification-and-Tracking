"""Unit tests for covering-disk chip area fraction and the 50% in-frame gate."""

import math

import numpy as np
from analysis.lib.optics import Optics
from analysis.lib.tracking import disk_frac_in_rect, disk_half_in_chip


def test_disk_frac_full_when_centered_and_small() -> None:
    """A small disk on boresight is entirely inside a wide rectangle."""
    frac = disk_frac_in_rect(1.0, 0.0, 0.0, -12.0, 12.0, -12.0, 12.0)
    assert float(frac) > 0.999


def test_disk_frac_half_when_origin_on_edge() -> None:
    """A small disk centered on a rectangle edge is half inside."""
    frac = disk_frac_in_rect(1.0, 0.0, 12.0, -12.0, 12.0, -12.0, 12.0)
    assert abs(float(frac) - 0.5) < 0.02


def test_disk_frac_zero_when_outside_by_more_than_r() -> None:
    """A disk whose center is more than R outside the rectangle has fraction 0."""
    frac = disk_frac_in_rect(1.0, 0.0, 20.0, -12.0, 12.0, -12.0, 12.0)
    assert float(frac) < 1.0e-9


def test_disk_frac_point_origin_inside() -> None:
    """R = 0 is a point: inside the rectangle is fraction 1."""
    frac = disk_frac_in_rect(0.0, 0.0, 0.0, -12.0, 12.0, -12.0, 12.0)
    assert float(frac) == 1.0


def test_disk_frac_point_origin_outside() -> None:
    """R = 0 outside the rectangle is fraction 0."""
    frac = disk_frac_in_rect(0.0, 0.0, 20.0, -12.0, 12.0, -12.0, 12.0)
    assert float(frac) == 0.0


def test_disk_frac_quarter_disk_in_first_quadrant_box() -> None:
    """[0, R] x [0, R] contains a quarter of a disk at the origin."""
    radius = 2.0
    frac = disk_frac_in_rect(radius, 0.0, 0.0, 0.0, radius, 0.0, radius)
    assert abs(float(frac) - 0.25) < 1.0e-6


def test_disk_frac_matches_half_plane_area() -> None:
    """A vertical cut through the center leaves half the disk."""
    radius = 3.0
    frac = disk_frac_in_rect(radius, 0.0, 0.0, -10.0, 10.0, -10.0, 0.0)
    assert abs(float(frac) - 0.5) < 1.0e-6


def _optics() -> Optics:
    """Return a square FOV with half-angle atan(12/400) so w = 12 km at 400 km slant."""
    half_deg = math.degrees(math.atan(12.0 / 400.0))
    full = 2.0 * half_deg
    return Optics(
        fov_az_raw_deg=full,
        fov_el_raw_deg=full,
        fov_az_deg=full,
        fov_el_deg=full,
        ifov_deg=0.001,
        sensor_width_mm=8.0,
        sensor_height_mm=7.0,
        datasheet_hfov_deg=3.36,
        computed_2_3_hfov_deg=3.36,
    )


def test_disk_half_one_axis_true_on_boresight() -> None:
    """One-axis half-disk gate passes when the origin is on boresight."""
    optics = _optics()
    az = np.array([0.0])
    slant = np.array([400.0])
    assert bool(disk_half_in_chip(az, slant, 2.0, optics, boresight_origin=False)[0])


def test_disk_half_one_axis_false_when_origin_far() -> None:
    """One-axis half-disk gate fails when the origin is well outside the chip."""
    optics = _optics()
    az = np.array([math.degrees(math.atan(20.0 / 400.0))])
    slant = np.array([400.0])
    assert not bool(disk_half_in_chip(az, slant, 2.0, optics, boresight_origin=False)[0])


def test_disk_half_two_axis_origin_outside_box_is_out() -> None:
    """Two-axis is out when the origin is outside the keep-out box."""
    optics = _optics()
    az = np.array([11.0])
    slant = np.array([400.0])
    assert not bool(
        disk_half_in_chip(az, slant, 2.0, optics, boresight_origin=True, az_box_deg=10.0)[0]
    )


def test_disk_half_two_axis_origin_in_box_passes() -> None:
    """Two-axis half-disk gate passes when the origin is in the box and R is small."""
    optics = _optics()
    az = np.array([5.0])
    slant = np.array([400.0])
    assert bool(
        disk_half_in_chip(az, slant, 2.0, optics, boresight_origin=True, az_box_deg=10.0)[0]
    )
