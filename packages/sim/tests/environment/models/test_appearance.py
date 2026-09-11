"""Tests for OracleMask rasterization bounds."""

from __future__ import annotations

import numpy as np
from flight.hal.interfaces.ephemeris import IssState
from flight.libs.config import SensorConfig
from flight.payload.gimbal.intersect import CameraGeometry
from sim.environment.models.appearance import OracleMask
from sim.environment.records import (
    LookAngles,
    PlumeState,
    SceneGeometry,
    ShutterPose,
    camera_from_sensor,
)


def _geom(centroid: tuple[float, float] | None) -> SceneGeometry:
    """Build a SceneGeometry that only the mask painter reads."""
    return SceneGeometry(
        iss=IssState(r_m=(0.0, 0.0, 0.0), v_m_s=(0.0, 0.0, 0.0), epoch_utc_s=0.0),
        plume=PlumeState(
            frame="ecef",
            cog_ecef_m=(0.0, 0.0, 0.0),
            centroid_band_px=centroid,
            along_sigma_m=0.0,
            cross_sigma_m=0.0,
            height_proxy_m=2000.0,
        ),
        look=LookAngles(0.0, 0.0, 0.0, 0.0, 0.0, False),
        centroid_band_px=centroid,
    )


def _paint(camera: CameraGeometry, centroid: tuple[float, float]) -> np.ndarray:
    """Return the oracle mask for a centroid."""
    feed = OracleMask().render_feed(
        _geom(centroid),
        ShutterPose(0.0, 0.0, 13.0, 0.0),
        camera,
        np.random.default_rng(0),
    )
    assert feed.mask is not None
    return feed.mask


def test_oracle_mask_off_frame_left_is_empty() -> None:
    """A centroid left of the chip does not paint a border square."""
    camera = camera_from_sensor(SensorConfig())
    mask = _paint(camera, (-100.0, 100.0))
    assert float(np.max(mask)) == 0.0


def test_oracle_mask_off_frame_right_is_empty() -> None:
    """A centroid right of the chip leaves the mask empty."""
    camera = camera_from_sensor(SensorConfig())
    mask = _paint(camera, (float(camera.width_px) + 100.0, 100.0))
    assert float(np.max(mask)) == 0.0


def test_oracle_mask_partial_left_is_cropped() -> None:
    """A near-left centroid paints only the on-chip part of the square."""
    camera = camera_from_sensor(SensorConfig())
    mask = _paint(camera, (10.0, 100.0))
    assert float(np.sum(mask[75:125, 0:35])) == 35.0 * 50.0
    assert float(np.sum(mask[:, 35:])) == 0.0
