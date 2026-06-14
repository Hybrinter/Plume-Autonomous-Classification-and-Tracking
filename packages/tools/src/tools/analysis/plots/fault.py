"""FDIR figures: SAFE latch + reason, mode changes, per-subsystem watchdog, per-code faults.

Satisfies: REQ-OBS-SIL-001.
"""

from __future__ import annotations

# third-party
import pandas as pd

# internal
from tools.analysis.datapoints import MONITORED
from tools.analysis.plots import common
from tools.analysis.plots.common import LabeledFigure

# The fault codes worth charting per-step (the SAFE-triggering set plus the loud non-SAFE ones).
_FAULT_CODES = (
    "THERMAL_OVER_LIMIT",
    "POWER_OVER_LIMIT",
    "GIMBAL_RUNAWAY",
    "WATCHDOG_EXPIRE",
    "MODEL_CORRUPT",
    "PROCESS_DIED",
    "STORAGE_FULL",
    "COMMAND_UNROUTABLE",
)


def build(wide: pd.DataFrame) -> list[LabeledFigure]:
    """Build the FDIR figures from the fault wide frame."""
    miss = [f"fault.miss.{subsystem}" for subsystem in MONITORED]
    age = [f"fault.heartbeat_age.{subsystem}" for subsystem in MONITORED]
    codes = [f"fault.code.{code}" for code in _FAULT_CODES]
    candidates = [
        common.categorical_timeline(
            wide, "fault.safe_reason", name="fault_safe_reason", title="Latched SAFE reason"
        ),
        common.line_panel(
            wide,
            [
                "fault.safe_latched",
                "fault.event_count",
                "fault.mode_change_count",
                "fault.safety_active_faults",
            ],
            name="fault_safety",
            title="SAFE latch + fault/mode-change activity",
            ylabel="count",
        ),
        common.line_panel(
            wide,
            miss,
            name="fault_watchdog_miss",
            title="Per-subsystem watchdog miss count",
            ylabel="consecutive misses",
        ),
        common.line_panel(
            wide,
            age,
            name="fault_heartbeat_age",
            title="Per-subsystem heartbeat age",
            ylabel="seconds",
        ),
        common.stacked_counts(
            wide, codes, name="fault_codes", title="Per-step fault events by code"
        ),
        common.cumulative_lines(
            wide,
            ["fault.event_count", "fault.mode_change_count"],
            name="fault_cumulative",
            title="Cumulative fault events + mode changes",
        ),
    ]
    return [figure for figure in candidates if figure is not None]
