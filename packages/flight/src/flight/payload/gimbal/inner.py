"""Inner elevation rate loop: PI + computed torque (pure, SI).

Implements the inner law: rate error into PI acceleration, then tau = J_hat * v +
B_hat * y_m, clipped to tau_max. The integrator freezes on torque clip or a
hardware travel stop.

Satisfies: REQ-AIML-GIMB-002, REQ-GIMB-HIGH-001.
"""

from __future__ import annotations

# stdlib
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class InnerResult:
    """One inner-loop output.

    Attributes:
        tau_nm: Clipped torque command, N·m.
        integrator: Updated PI integrator state (rad/s * s).
        v_rad_s2: Commanded angular acceleration, rad/s².
        clipped: True when the unsaturated torque exceeded tau_max.
    """

    tau_nm: float
    integrator: float
    v_rad_s2: float
    clipped: bool


def inner_step(
    r_rad_s: float,
    y_m: float,
    integrator: float,
    dt_s: float,
    j_hat: float,
    b_hat: float,
    kp: float,
    ki: float,
    tau_max_nm: float,
    stopped: bool,
) -> InnerResult:
    """Advance the inner PI by one inner period.

    Inputs:
        r_rad_s: Rate reference from the outer or position loop, rad/s.
        y_m: Encoder-rate estimate, rad/s.
        integrator: PI integrator state.
        dt_s: Inner period, seconds.
        j_hat, b_hat: Plant copies used in the computed-torque term.
        kp, ki: PI gains (1/s and 1/s² on rad/s).
        tau_max_nm: Torque clip, N·m.
        stopped: True when the encoder is on a hardware travel stop.

    Outputs:
        InnerResult: Clipped torque, updated integrator, acceleration, clip flag.
    """
    err = r_rad_s - y_m
    v = kp * err + ki * integrator
    tau_unsat = j_hat * v + b_hat * y_m
    if tau_unsat > tau_max_nm:
        tau = tau_max_nm
    elif tau_unsat < -tau_max_nm:
        tau = -tau_max_nm
    else:
        tau = tau_unsat
    clipped = abs(tau_unsat) > tau_max_nm + 1e-15
    new_i = integrator
    if (not clipped) and (not stopped) and dt_s > 0.0:
        new_i = integrator + err * dt_s
    return InnerResult(tau_nm=tau, integrator=new_i, v_rad_s2=v, clipped=clipped)
