"""Unit tests for analysis.lib.tracking disk any-part and science window."""

import numpy as np
from analysis.lib.tracking import az_interval_overlaps, disk_any_part_in_stop


def test_az_interval_overlaps_when_span_crosses_boresight() -> None:
    """A disk spanning -2 to +2 deg overlaps a +/-1.6 deg FOV."""
    a = np.array([-2.0])
    b = np.array([2.0])
    assert bool(az_interval_overlaps(a, b, 1.6)[0])


def test_az_interval_no_overlap_when_both_edges_off_fov() -> None:
    """A disk entirely at +3 to +5 deg does not overlap +/-1.6 deg."""
    a = np.array([3.0])
    b = np.array([5.0])
    assert not bool(az_interval_overlaps(a, b, 1.6)[0])


def test_disk_any_part_true_when_origin_in_fov() -> None:
    """Origin in the FOV is enough for any-part, even if edges stick out."""
    origin = np.array([0.0])
    plus = np.array([2.0])
    minus = np.array([-2.0])
    assert bool(disk_any_part_in_stop(origin, plus, minus, 1.6)[0])


def test_disk_any_part_two_axis_box() -> None:
    """A disk that only grazes the keep-out box still counts as any-part."""
    origin = np.array([11.0])
    plus = np.array([9.5])
    minus = np.array([12.0])
    assert bool(disk_any_part_in_stop(origin, plus, minus, 10.0)[0])
    assert not bool(disk_any_part_in_stop(origin, plus, minus, 8.0)[0])
