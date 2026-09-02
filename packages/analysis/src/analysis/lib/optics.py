"""Thin-lens FOV from pixel pitch, array size, and focal length.

A distortion shrink is applied after the thin-lens FOV so a study can take
the most restrictive usable field. Datasheet 2/3-inch HFOV is stored only
as a cross-check against the computed 8.8 mm value; it is not the camera FOV.

Contains:
  - OpticsSpec: pixel / array / lens inputs.
  - Optics: computed raw and usable FOV.
  - fov_deg / ifov_deg_per_px / band_gsd_along_m: thin-lens helpers.
  - build_optics: OpticsSpec -> Optics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class OpticsSpec:
    """Pixel array and lens used to compute usable FOV.

    Attributes:
        pixel_um: Pixel pitch in micrometres.
        n_lateral_px: Pixel count on the cross-track (azimuth) axis.
        n_along_px: Pixel count on the along-track (elevation) axis.
        focal_length_mm: Lens focal length in millimetres.
        datasheet_2_3_width_mm: 2/3-inch format width used only as a check.
        datasheet_hfov_2_3_deg: Published HFOV for that 2/3-inch width.
        lens_distortion_pct: Max distortion applied as a shrink of usable FOV.
    """

    pixel_um: float
    n_lateral_px: int
    n_along_px: int
    focal_length_mm: float
    datasheet_2_3_width_mm: float
    datasheet_hfov_2_3_deg: float
    lens_distortion_pct: float


@dataclass(frozen=True)
class Optics:
    """Computed sensor FOV after the distortion shrink.

    Attributes:
        fov_az_raw_deg: Thin-lens azimuth FOV before distortion shrink.
        fov_el_raw_deg: Thin-lens elevation FOV before distortion shrink.
        fov_az_deg: Usable azimuth FOV after distortion shrink.
        fov_el_deg: Usable elevation FOV after distortion shrink.
        ifov_deg: Instantaneous FOV per mosaic pixel, degrees.
        ifov_band_deg: IFOV per 2x2 band-plane pixel (property, 2 * ifov_deg).
        sensor_width_mm: Active width (lateral).
        sensor_height_mm: Active height (along-track).
        datasheet_hfov_deg: Published 2/3-inch HFOV (not the camera FOV).
        computed_2_3_hfov_deg: Thin-lens HFOV for the 8.8 mm 2/3-inch width.
    """

    fov_az_raw_deg: float
    fov_el_raw_deg: float
    fov_az_deg: float
    fov_el_deg: float
    ifov_deg: float
    sensor_width_mm: float
    sensor_height_mm: float
    datasheet_hfov_deg: float
    computed_2_3_hfov_deg: float

    @property
    def half_az_deg(self) -> float:
        """Return half the usable azimuth FOV in degrees."""
        return 0.5 * self.fov_az_deg

    @property
    def half_el_deg(self) -> float:
        """Return half the usable elevation FOV in degrees."""
        return 0.5 * self.fov_el_deg

    @property
    def ifov_band_deg(self) -> float:
        """Return IFOV per 2x2-demosaiced band-plane pixel, degrees.

        The mosaic is a 2x2 CFA, so each band plane is half the linear
        resolution of the detector. Band-plane IFOV is twice the mosaic IFOV.
        """
        return 2.0 * self.ifov_deg


def fov_deg(n_px: int, pixel_um: float, fl_mm: float) -> float:
    """Return thin-lens full FOV in degrees for ``n_px`` pixels.

    Args:
        n_px: Pixel count along the axis.
        pixel_um: Pixel pitch in micrometres.
        fl_mm: Focal length in millimetres.

    Returns:
        Full angular FOV in degrees.
    """
    half_mm = (n_px * pixel_um / 1000.0) / 2.0
    return 2.0 * math.degrees(math.atan(half_mm / fl_mm))


def band_gsd_along_m(ifov_band_deg: float, slant_km: float, incidence_deg: float) -> float:
    """Return along-track band-cell GSD in metres.

    Ground-projected GSD is ``ifov * slant / cos(incidence)``. At the limb
    incidence approaches 90 deg and GSD diverges.

    Args:
        ifov_band_deg: Band-plane IFOV in degrees per pixel.
        slant_km: Slant range in kilometres.
        incidence_deg: Earth emission angle in degrees.

    Returns:
        Along-track GSD in metres. Inf when incidence is 90 deg.
    """
    cos_i = math.cos(math.radians(incidence_deg))
    if cos_i <= 1e-6:
        return math.inf
    return math.radians(ifov_band_deg) * slant_km * 1000.0 / cos_i


def ifov_deg_per_px(pixel_um: float, fl_mm: float) -> float:
    """Return instantaneous FOV in degrees per pixel.

    Args:
        pixel_um: Pixel pitch in micrometres.
        fl_mm: Focal length in millimetres.

    Returns:
        IFOV in degrees per pixel.
    """
    return math.degrees(math.atan((pixel_um / 1000.0) / fl_mm))


def build_optics(spec: OpticsSpec) -> Optics:
    """Build usable FOV from an optics spec, applying the distortion shrink.

    Args:
        spec: Pixel array and lens.

    Returns:
        Computed Optics. The 2/3-inch sheet HFOV is a check only.
    """
    raw_az = fov_deg(spec.n_lateral_px, spec.pixel_um, spec.focal_length_mm)
    raw_el = fov_deg(spec.n_along_px, spec.pixel_um, spec.focal_length_mm)
    shrink = 1.0 - spec.lens_distortion_pct / 100.0
    computed_23 = 2.0 * math.degrees(
        math.atan((spec.datasheet_2_3_width_mm / 2.0) / spec.focal_length_mm)
    )
    return Optics(
        fov_az_raw_deg=raw_az,
        fov_el_raw_deg=raw_el,
        fov_az_deg=raw_az * shrink,
        fov_el_deg=raw_el * shrink,
        ifov_deg=ifov_deg_per_px(spec.pixel_um, spec.focal_length_mm),
        sensor_width_mm=spec.n_lateral_px * spec.pixel_um / 1000.0,
        sensor_height_mm=spec.n_along_px * spec.pixel_um / 1000.0,
        datasheet_hfov_deg=spec.datasheet_hfov_2_3_deg,
        computed_2_3_hfov_deg=computed_23,
    )
