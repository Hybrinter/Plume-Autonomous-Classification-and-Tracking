"""Payload figures: gimbal FSM, pointing, torque, residual filter, science output.

Satisfies: REQ-OBS-SIL-001.
"""

from __future__ import annotations

# third-party
import pandas as pd

# internal
from tools.analysis.plots import common
from tools.analysis.plots.common import LabeledFigure


def build(wide: pd.DataFrame) -> list[LabeledFigure]:
    """Build the payload figures from the payload wide frame."""
    candidates = [
        common.categorical_timeline(
            wide, "payload.gimbal_state", name="payload_fsm", title="Gimbal arbiter FSM state"
        ),
        common.line_panel(
            wide,
            [
                "payload.gimbal_el_true_deg",
                "payload.gimbal_el_meas_deg",
            ],
            name="payload_pointing",
            title="Gimbal elevation (truth vs measured)",
            ylabel="deg",
        ),
        common.line_panel(
            wide,
            [
                "payload.r_rad_s",
                "payload.y_m",
                "payload.gimbal_omega_rad_s",
            ],
            name="payload_rates",
            title="Rate reference vs encoder vs plant",
            ylabel="rad/s",
        ),
        common.line_panel(
            wide,
            [
                "payload.e_hat",
                "payload.omega_t_res",
                "payload.omega_t_nom",
            ],
            name="payload_residual_state",
            title="Residual filter and predictor",
            ylabel="rad, rad/s",
        ),
        common.line_panel(
            wide,
            [
                "payload.residual_p00",
                "payload.residual_p11",
                "payload.residual_p_trace",
            ],
            name="payload_residual_cov",
            title="Residual covariance diagonal + trace",
            ylabel="variance",
        ),
        common.line_panel(
            wide,
            [
                "payload.tau_nm",
                "payload.gimbal_tau_nm",
            ],
            name="payload_torque",
            title="Inner torque command",
            ylabel="N*m",
        ),
        common.line_panel(
            wide,
            [
                "payload.miss_count",
                "payload.tracked_blobs",
            ],
            name="payload_safety_counters",
            title="Tracking counters",
            ylabel="count",
        ),
        common.line_panel(
            wide,
            [
                "payload.motion_inhibited",
                "payload.stow_switch",
                "payload.is_tracking",
                "payload.vision_accepted",
            ],
            name="payload_flags",
            title="Payload state flags",
            ylabel="flag (0/1)",
        ),
        common.stacked_counts(
            wide,
            [
                "payload.inference_count",
                "payload.gimbal_command_count",
                "payload.product_ref_count",
                "payload.fault_count",
            ],
            name="payload_output",
            title="Per-step payload output (inference/command/product/fault)",
        ),
    ]
    return [figure for figure in candidates if figure is not None]
