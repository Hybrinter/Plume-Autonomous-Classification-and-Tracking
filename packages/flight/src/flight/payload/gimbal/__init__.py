"""Payload gimbal control: pointing FSM, inner/outer laws, and safety gates (pure).

arbiter -- TRACKING / REWIND / SAFE mode selection;
inner -- PI + computed torque;
outer -- scene match, smear clip, and RateDecision;
position -- STOW/HOME/GOTO rate into the inner PI;
rate_fit -- causal polynomial encoder-rate estimator;
intersect -- pinhole CoG and boresight height-ellipsoid intersect;
predictor -- co-rotating elevation and unactuated azimuth rates;
geo -- mount / LVLH / WGS-84 helpers;
pointing -- pinhole boresight error;
request -- typed pose command from the pure core;
safety -- confidence and area gates.
"""

from flight.payload.gimbal.arbiter import ArbiterState, GimbalArbiter
from flight.payload.gimbal.inner import InnerResult, inner_step
from flight.payload.gimbal.integrity import IntegrityResult, check_integrity, lock_hold_rate
from flight.payload.gimbal.intersect import (
    CameraGeometry,
    RayHit,
    intersect_boresight,
    intersect_cog,
)
from flight.payload.gimbal.outer import RateDecision, clip_rate, outer_rate, smear_cap_rad_s
from flight.payload.gimbal.pointing import (
    boresight_error_deg,
    pinhole_error_rad,
    target_displacement_px,
)
from flight.payload.gimbal.position import position_rate
from flight.payload.gimbal.predictor import LosPrediction, predict_los
from flight.payload.gimbal.rate_fit import fit_rate, fit_rate_timed
from flight.payload.gimbal.request import GimbalRequest
from flight.payload.gimbal.safety import apply_confidence_gate, apply_min_area_gate

__all__ = [
    "ArbiterState",
    "CameraGeometry",
    "GimbalArbiter",
    "GimbalRequest",
    "InnerResult",
    "IntegrityResult",
    "LosPrediction",
    "RateDecision",
    "RayHit",
    "apply_confidence_gate",
    "apply_min_area_gate",
    "boresight_error_deg",
    "check_integrity",
    "clip_rate",
    "fit_rate",
    "fit_rate_timed",
    "inner_step",
    "intersect_boresight",
    "intersect_cog",
    "lock_hold_rate",
    "outer_rate",
    "pinhole_error_rad",
    "position_rate",
    "predict_los",
    "smear_cap_rad_s",
    "target_displacement_px",
]
