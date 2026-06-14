"""Mechanical figures: launch-lock state + interlock telemetry and observed gimbal motion.

Satisfies: REQ-OBS-SIL-001.
"""

from __future__ import annotations

# third-party
import pandas as pd

# internal
from tools.analysis.plots import common
from tools.analysis.plots.common import LabeledFigure


def build(wide: pd.DataFrame) -> list[LabeledFigure]:
    """Build the mechanical figures from the mechanical wide frame."""
    candidates = [
        common.categorical_timeline(
            wide,
            "mechanical.launch_lock_state",
            name="mechanical_lock_state",
            title="Launch-lock state",
        ),
        common.line_panel(
            wide,
            ["mechanical.launch_lock_engaged"],
            name="mechanical_lock_engaged",
            title="Launch-lock engaged",
            ylabel="engaged (0/1)",
        ),
        common.stacked_counts(
            wide,
            [
                "mechanical.lock_state_msg_count",
                "mechanical.lock_fault_count",
                "mechanical.gimbal_cmd_observed",
            ],
            name="mechanical_activity",
            title="Per-step lock telemetry + observed gimbal motion",
        ),
        common.cumulative_lines(
            wide,
            ["mechanical.lock_state_msg_count"],
            name="mechanical_lock_cumulative",
            title="Cumulative launch-lock state publications",
        ),
    ]
    return [figure for figure in candidates if figure is not None]
