"""Small image-forming oracle for target-relative exposure blur.

It renders the plume over exposure substeps using simulated gimbal motion.
The controller-facing result is only an image centroid.  Truth motion and
per-region blur diagnostics never enter the controller.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class PlumeRegion:
    """One Gaussian region at mid-exposure image coordinates."""

    name: str
    center_x_px: float
    center_y_px: float
    velocity_x_px_s: float
    velocity_y_px_s: float
    sigma_px: float
    apparent_area_px: float
    intensity: float = 1.0


@dataclass(frozen=True, slots=True)
class LocalBlur:
    """Per-region target-relative blur and sharp/detected-area contribution."""

    name: str
    displacement_px: float
    detected: bool
    sharp_detected_area_px: float


@dataclass(frozen=True, slots=True)
class ExposureImage:
    """Integrated image, controller observation, and truth-only quality metrics."""

    image: np.ndarray
    centroid_x_px: float | None
    centroid_y_px: float | None
    local_blur: tuple[LocalBlur, ...]

    @property
    def sharp_detected_area_px(self) -> float:
        """Return total visible area that was both detected and within blur budget."""
        return sum(item.sharp_detected_area_px for item in self.local_blur)


def integrate_exposure(
    regions: tuple[PlumeRegion, ...],
    shape_px: tuple[int, int],
    exposure_s: float,
    gimbal_rate_rad_s: float,
    ifov_rad_per_px: float,
    smear_budget_px: float,
    substeps: int = 17,
) -> ExposureImage:
    """Project and integrate moving regions over a finite exposure interval.

    Image +y is elevation error for this analysis.  Gimbal motion therefore
    subtracts from each region's apparent vertical pixel rate.  ``substeps``
    must be odd so a sample lies at shutter midpoint.
    """
    if exposure_s <= 0.0 or ifov_rad_per_px <= 0.0:
        raise ValueError("exposure and IFOV must be positive")
    if substeps < 3 or substeps % 2 == 0:
        raise ValueError("substeps must be odd and at least three")
    height, width = shape_px
    yy, xx = np.mgrid[0:height, 0:width]
    image = np.zeros(shape_px, dtype=np.float64)
    local: list[LocalBlur] = []
    offsets_s = np.linspace(-0.5 * exposure_s, 0.5 * exposure_s, substeps)
    for region in regions:
        apparent_y_rate = region.velocity_y_px_s - gimbal_rate_rad_s / ifov_rad_per_px
        for offset_s in offsets_s:
            x = region.center_x_px + region.velocity_x_px_s * offset_s
            y = region.center_y_px + apparent_y_rate * offset_s
            radius_sq = (xx - x) ** 2 + (yy - y) ** 2
            image += region.intensity * np.exp(-0.5 * radius_sq / region.sigma_px**2) / substeps
        displacement = exposure_s * float(np.hypot(region.velocity_x_px_s, apparent_y_rate))
        visible = (
            -3.0 * region.sigma_px <= region.center_x_px < width + 3.0 * region.sigma_px
            and -3.0 * region.sigma_px <= region.center_y_px < height + 3.0 * region.sigma_px
        )
        local.append(
            LocalBlur(
                name=region.name,
                displacement_px=displacement,
                detected=visible,
                sharp_detected_area_px=(
                    region.apparent_area_px if visible and displacement <= smear_budget_px else 0.0
                ),
            )
        )
    total = float(image.sum())
    if total <= 0.0:
        return ExposureImage(
            image=image,
            centroid_x_px=None,
            centroid_y_px=None,
            local_blur=tuple(local),
        )
    return ExposureImage(
        image=image,
        centroid_x_px=float((image * xx).sum() / total),
        centroid_y_px=float((image * yy).sum() / total),
        local_blur=tuple(local),
    )
