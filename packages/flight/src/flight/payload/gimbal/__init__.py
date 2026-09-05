"""Payload gimbal control: pointing FSM, inner/outer laws, and safety gates (pure).

arbiter -- TRACKING / REWIND / SAFE mode selection;
inner -- PI + computed torque;
outer -- residual feedforward and smear clip;
position -- STOW/HOME/GOTO rate into the inner PI;
rate_fit -- causal polynomial encoder-rate estimator;
intersect -- pinhole CoG Earth intersect;
predictor -- co-rotating elevation rate;
geo -- mount / LVLH / WGS-84 helpers;
pointing -- pinhole boresight error;
request -- typed pose command from the pure core;
safety -- confidence and area gates.
"""

from flight.payload.gimbal.arbiter import ArbiterState, GimbalArbiter
from flight.payload.gimbal.inner import InnerResult, inner_step
from flight.payload.gimbal.intersect import IntersectResult, intersect_cog
from flight.payload.gimbal.outer import clip_rate, outer_rate, smear_cap_rad_s
from flight.payload.gimbal.pointing import (
    boresight_error_deg,
    pinhole_error_rad,
    target_displacement_px,
)
from flight.payload.gimbal.position import position_rate
from flight.payload.gimbal.predictor import predict_los
from flight.payload.gimbal.rate_fit import fit_rate
from flight.payload.gimbal.request import GimbalRequest
from flight.payload.gimbal.safety import apply_confidence_gate, apply_min_area_gate

__all__ = [
    "ArbiterState",
    "GimbalArbiter",
    "GimbalRequest",
    "InnerResult",
    "IntersectResult",
    "apply_confidence_gate",
    "apply_min_area_gate",
    "boresight_error_deg",
    "clip_rate",
    "fit_rate",
    "inner_step",
    "intersect_cog",
    "outer_rate",
    "pinhole_error_rad",
    "position_rate",
    "predict_los",
    "smear_cap_rad_s",
    "target_displacement_px",
]
