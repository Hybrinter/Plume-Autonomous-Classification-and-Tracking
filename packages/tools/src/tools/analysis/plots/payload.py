"""Payload figures: gimbal FSM, pointing, control rates, tracking estimators, science output.

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
                "payload.gimbal_az_true_deg",
                "payload.gimbal_el_true_deg",
                "payload.gimbal_az_meas_deg",
                "payload.gimbal_el_meas_deg",
            ],
            name="payload_pointing",
            title="Gimbal pointing (truth vs measured)",
            ylabel="deg",
        ),
        common.line_panel(
            wide,
            [
                "payload.commanded_az_rate_deg_s",
                "payload.commanded_el_rate_deg_s",
                "payload.gimbal_rate_az_deg_s",
                "payload.gimbal_rate_el_deg_s",
            ],
            name="payload_rates",
            title="Commanded vs driver gimbal rates",
            ylabel="deg/s",
        ),
        common.line_panel(
            wide,
            [
                "payload.kalman_az_err",
                "payload.kalman_el_err",
                "payload.kalman_az_vel",
                "payload.kalman_el_vel",
            ],
            name="payload_kalman_state",
            title="Kalman state estimate",
            ylabel="deg, deg/s",
        ),
        common.line_panel(
            wide,
            [
                "payload.kalman_p00",
                "payload.kalman_p11",
                "payload.kalman_p22",
                "payload.kalman_p33",
                "payload.kalman_p_trace",
            ],
            name="payload_kalman_cov",
            title="Kalman covariance diagonal + trace",
            ylabel="variance",
        ),
        common.line_panel(
            wide,
            ["payload.ema_centroid_x", "payload.ema_centroid_y", "payload.ema_error_mag"],
            name="payload_ema",
            title="EMA boresight error",
            ylabel="deg",
        ),
        common.line_panel(
            wide,
            [
                "payload.miss_count",
                "payload.deadband_strikes",
                "payload.runaway_strikes",
                "payload.tracked_blobs",
            ],
            name="payload_safety_counters",
            title="Tracking + safety counters",
            ylabel="count",
        ),
        common.line_panel(
            wide,
            [
                "payload.motion_inhibited",
                "payload.stow_switch",
                "payload.is_tracking",
                "payload.ema_initialized",
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
