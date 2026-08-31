"""Payload gimbal control: the pointing FSM, control law, and safety gates (pure).

arbiter -- the IDLE/ACQUIRING/TRACKING/SCAN/SAFE FSM and command generation;
lqr -- discrete-LQR control law; pointing -- boresight-relative angular error math;
rate_servo -- inner rate PI with computed-torque decoupling;
request -- typed command value from the pure core;
runaway -- encoder-divergence runaway monitor;
safety -- confidence/area/deadband/rate gates.
"""

from flight.payload.gimbal.arbiter import ArbiterState, GimbalArbiter
from flight.payload.gimbal.lqr import LqrController, compute_control
from flight.payload.gimbal.pointing import boresight_error_deg, target_displacement_px
from flight.payload.gimbal.rate_servo import (
    INITIAL_RATE_SERVO_STATE,
    RateServoState,
    reset_servo,
)
from flight.payload.gimbal.rate_servo import (
    step as rate_servo_step,
)
from flight.payload.gimbal.request import GimbalRequest
from flight.payload.gimbal.runaway import INITIAL_RUNAWAY_STATE, RunawayState, check_runaway
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
    "INITIAL_RATE_SERVO_STATE",
    "INITIAL_RUNAWAY_STATE",
    "LqrController",
    "RateServoState",
    "RunawayState",
    "apply_confidence_gate",
    "apply_min_area_gate",
    "boresight_error_deg",
    "check_deadband",
    "check_rate_limit",
    "check_runaway",
    "compute_control",
    "rate_servo_step",
    "reset_servo",
    "target_displacement_px",
]
