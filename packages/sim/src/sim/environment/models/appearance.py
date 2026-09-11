"""Appearance models: oracle mask (and later geometric / radiometric mosaics)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

# third-party
import numpy as np

# internal
from flight.payload.gimbal.intersect import CameraGeometry

from sim.environment.records import DriverFeed, SceneGeometry, ShutterPose

_MASK_HALF = 25  # 50x50 square like sim.scene.plume.plume_detector


@runtime_checkable
class AppearanceModel(Protocol):
    """Render a driver feed from scene geometry."""

    def render_feed(
        self,
        geom: SceneGeometry,
        shutter: ShutterPose,
        camera: CameraGeometry,
        rng: np.random.Generator,
    ) -> DriverFeed:
        """Return mosaic and/or mask. None fields mean the bind must not push."""
        ...


@dataclass(frozen=True, slots=True)
class OracleMask:
    """Paint a unit-probability square at the projected CoG. No mosaic."""

    def render_feed(
        self,
        geom: SceneGeometry,
        shutter: ShutterPose,
        camera: CameraGeometry,
        rng: np.random.Generator,
    ) -> DriverFeed:
        """Return a band-plane mask. Mosaic stays None (keep SIL replay frames)."""
        del shutter, rng
        centroid = geom.centroid_band_px
        mask = np.zeros((camera.height_px, camera.width_px), dtype=np.float32)
        if centroid is not None:
            u_c, v_c = centroid
            u0 = max(0, int(round(u_c)) - _MASK_HALF)
            v0 = max(0, int(round(v_c)) - _MASK_HALF)
            u1 = min(camera.width_px, u0 + 2 * _MASK_HALF)
            v1 = min(camera.height_px, v0 + 2 * _MASK_HALF)
            mask[v0:v1, u0:u1] = 1.0
        return DriverFeed(mosaic=None, mask=mask)
