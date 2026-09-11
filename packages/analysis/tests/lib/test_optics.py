"""Facade checks: OpticsSpec maps onto sim CameraGeometry."""

from analysis.lib.optics import camera_from_optics_spec
from analysis.studies.single_axis_vs_dual_axis_gimbal.assumptions import OPTICS_SPEC
from flight.libs.config import SensorConfig
from sim.environment.records import camera_from_sensor


def test_camera_from_optics_spec_matches_sensor_helper_when_sizes_align() -> None:
    """Band-plane pitch is twice mosaic pitch, matching camera_from_sensor."""
    sensor = SensorConfig()
    from_spec = camera_from_optics_spec(OPTICS_SPEC)
    from_sensor = camera_from_sensor(sensor)
    assert from_spec.pixel_pitch_m == from_sensor.pixel_pitch_m
    assert from_spec.focal_length_m == from_sensor.focal_length_m
