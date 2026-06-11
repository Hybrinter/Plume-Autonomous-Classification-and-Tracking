"""Payload gimbal control: the pointing FSM, control law, and safety gates (pure).

arbiter -- the IDLE/ACQUIRING/TRACKING/SCAN/SAFE FSM and command generation;
lqr -- discrete-LQR control law; pointing -- boresight-relative angular error math;
request -- typed command value from the pure core;
safety -- confidence/area/deadband/rate gates.
"""

from flight.payload.gimbal.arbiter import ArbiterState, GimbalArbiter
from flight.payload.gimbal.lqr import LqrController, compute_control
from flight.payload.gimbal.pointing import boresight_error_deg, target_displacement_px
from flight.payload.gimbal.request import GimbalRequest
from flight.payload.gimbal.safety import (
    apply_confidence_gate,
    apply_min_area_gate,
    check_deadband,
    check_rate_limit,
)

__all__ = [
    "ArbiterState",
    "GimbalArbiter",
    "GimbalRequest",
    "LqrController",
    "apply_confidence_gate",
    "apply_min_area_gate",
    "boresight_error_deg",
    "check_deadband",
    "check_rate_limit",
    "compute_control",
    "target_displacement_px",
]
