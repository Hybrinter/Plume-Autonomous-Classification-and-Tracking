"""Model-deploy figures: lifecycle state, active version, staged/rollback flags, transitions.

Satisfies: REQ-OBS-SIL-001.
"""

from __future__ import annotations

# third-party
import pandas as pd

# internal
from tools.analysis.plots import common
from tools.analysis.plots.common import LabeledFigure


def build(wide: pd.DataFrame) -> list[LabeledFigure]:
    """Build the model-deploy figures from the model_deploy wide frame."""
    candidates = [
        common.categorical_timeline(
            wide, "model_deploy.state", name="model_state", title="Model deploy lifecycle state"
        ),
        common.categorical_timeline(
            wide,
            "model_deploy.active_version",
            name="model_active_version",
            title="Active model version",
        ),
        common.line_panel(
            wide,
            [
                "model_deploy.has_staged",
                "model_deploy.has_rollback",
                "model_deploy.active_is_factory",
            ],
            name="model_flags",
            title="Staged / rollback / factory flags",
            ylabel="flag (0/1)",
        ),
        common.line_panel(
            wide,
            ["model_deploy.staged_input_dims", "model_deploy.staged_output_dims"],
            name="model_staged_shape",
            title="Staged model shape rank",
            ylabel="rank",
        ),
        common.stacked_counts(
            wide,
            ["model_deploy.state_msg_count", "model_deploy.corrupt_fault_count"],
            name="model_activity",
            title="Per-step deploy transitions + corrupt faults",
        ),
        common.cumulative_lines(
            wide,
            ["model_deploy.state_msg_count"],
            name="model_state_cumulative",
            title="Cumulative deploy-state transitions",
        ),
    ]
    return [figure for figure in candidates if figure is not None]
