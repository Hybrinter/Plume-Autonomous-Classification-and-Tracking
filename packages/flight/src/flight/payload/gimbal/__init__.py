"""Payload gimbal control: pointing FSM, inner/outer laws, and safety gates (pure).

inner -- PI + computed torque;
outer -- smear clip, stopping governor, and RateDecision primitives;
scene -- CoG / boresight scene prediction entry points;
position -- STOW/HOME/GOTO rate into the inner PI;
rate_fit -- causal polynomial encoder-rate estimator;
intersect -- pinhole CoG and boresight height-ellipsoid intersect;
predictor -- co-rotating elevation and unactuated azimuth rates;
geo -- mount / LVLH / WGS-84 helpers;
pointing -- pinhole boresight error;
request -- typed mode-free control references;
safety -- confidence and area gates.
"""

from flight.payload.gimbal.inner import InnerResult, inner_step
from flight.payload.gimbal.integrity import IntegrityResult, check_integrity
from flight.payload.gimbal.intersect import (
    CameraGeometry,
    RayHit,
    intersect_boresight,
    intersect_cog,
)
from flight.payload.gimbal.outer import (
    RateDecision,
    clip_rate,
    rate_decision,
    smear_cap_rad_s,
)
from flight.payload.gimbal.pointing import (
    boresight_error_deg,
    pinhole_error_rad,
    target_displacement_px,
)
from flight.payload.gimbal.position import position_rate
from flight.payload.gimbal.predictor import LosPrediction, predict_los
from flight.payload.gimbal.rate_fit import fit_rate, fit_rate_timed
from flight.payload.gimbal.request import (
    ControlReference,
    InhibitReference,
    PoseReference,
    RateReference,
    StowReference,
    TravelEnvelope,
    validate_reference,
)
from flight.payload.gimbal.safety import apply_confidence_gate, apply_min_area_gate
from flight.payload.gimbal.scene import (
    SceneEstimate,
    SceneSource,
    boresight_scene,
    cog_scene,
)

__all__ = [
    "CameraGeometry",
    "ControlReference",
    "InhibitReference",
    "InnerResult",
    "IntegrityResult",
    "LosPrediction",
    "PoseReference",
    "RateDecision",
    "RateReference",
    "RayHit",
    "SceneEstimate",
    "SceneSource",
    "StowReference",
    "TravelEnvelope",
    "apply_confidence_gate",
    "apply_min_area_gate",
    "boresight_error_deg",
    "boresight_scene",
    "check_integrity",
    "clip_rate",
    "cog_scene",
    "fit_rate",
    "fit_rate_timed",
    "inner_step",
    "intersect_boresight",
    "intersect_cog",
    "pinhole_error_rad",
    "position_rate",
    "predict_los",
    "rate_decision",
    "smear_cap_rad_s",
    "target_displacement_px",
    "validate_reference",
]
