"""Locked assumptions for the single-axis vs dual-axis gimbal study.

These values are the 2026-09-01 lock: one-sided 90->30 deg window,
geocentric nadir, current ISS TLE, most restrictive FOV, design cluster
R = 5 km. Paths resolve to analysis/single_axis_vs_dual_axis_gimbal/.

Contains:
  - TLE / OPTICS_SPEC / GIMBAL_BOX and design-pass constants.
  - analysis_root / STUDY_DIR / DATA_DIR / CACHE_DIR / OUT_DIR.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from analysis.lib.look import GimbalBox, WindowMode
from analysis.lib.optics import OpticsSpec
from analysis.lib.orbit import IssTle

# Celestrak GP TLE, NORAD 25544, epoch 2026-09-01 (day 244.4985).
# 1 25544U 98067A   26244.49851261  .00003910  00000-0  79223-4 0  9993
# 2 25544  51.6312 282.3953 0005055  96.4740 263.6825 15.48958602583586
TLE = IssTle(
    epoch="2026-09-01",
    inclination_deg=51.6312,
    eccentricity=0.0005055,
    mean_motion_rev_per_day=15.48958602,
)

DESIGN_LAT_DEG = TLE.inclination_deg
PASS_LATS_DEG: tuple[float, ...] = (0.0, 20.0, 30.0, 40.0, 45.0, 50.0, TLE.inclination_deg)

GIMBAL_BOX = GimbalBox(
    az_box_deg=10.0,
    el_nadir_deg=90.0,
    el_limb_deg=30.0,
    window_mode=WindowMode.ONE_SIDED,
)
NADIR_OFFSET_DEG = 0.0

OPTICS_SPEC = OpticsSpec(
    pixel_um=3.45,
    n_lateral_px=2448,
    n_along_px=2048,
    focal_length_mm=150.0,
    datasheet_2_3_width_mm=8.8,
    datasheet_hfov_2_3_deg=3.36,
    lens_distortion_pct=0.66,
)

MAX_SLEW_DEG_S = 2.0

CLUSTER_R_KM = 5.0
DISK_RADII_KM: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0)
WIND_MPS: tuple[float, ...] = (0.0, 10.0, 20.0)
ORIGIN_OFFSETS_KM: tuple[float, ...] = (0.0, 5.0, 10.0, 20.0, 40.0, 70.0)

BAND_LATERAL_PX = OPTICS_SPEC.n_lateral_px // 2
BAND_ALONG_PX = OPTICS_SPEC.n_along_px // 2
SEG_FLOPS_256_G = 0.21
CLS_FLOPS_256_G = 0.110
SEG_LAT_256_MS = 2.3
CLS_LAT_256_MS = 3.4

GEOMETRY_DT_S = 0.05
INDUSTRY_PASS_DT_S = 0.1

PLUME_R_KM = 2.0
CLUSTER_EPS_KM = 8.0
MAX_CLUSTER_R_KM = 12.0
MIN_PLANT_R_KM = 0.4
LAT_BIN_DEG = 2.0
INVENTORY_YEAR = "2025"

GRID_R_KM: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0)


def analysis_root() -> Path:
    """Return the analysis/ workspace-member root (contains pyproject.toml).

    Returns:
        Path to analysis/.

    Raises:
        RuntimeError: If this file is not under the analysis member.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src" / "analysis").is_dir():
            return parent
    raise RuntimeError("could not find analysis/ member root")


STUDY_DIR = analysis_root() / "single_axis_vs_dual_axis_gimbal"
DATA_DIR = STUDY_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
OUT_DIR = STUDY_DIR / "outputs"


def grid_lats_deg() -> tuple[float, ...]:
    """Return |lat| sample points from 0 deg to inclination inclusive.

    Returns:
        26 uniformly spaced points from 0 to 50 deg, plus the TLE inclination.
    """
    return tuple([*np.linspace(0.0, 50.0, 26), TLE.inclination_deg])
