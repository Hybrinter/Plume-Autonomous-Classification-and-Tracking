"""System-rollup figures: mode timeline, SAFE latch, and gross message/fault throughput.

Satisfies: REQ-OBS-SIL-001.
"""

from __future__ import annotations

# third-party
import pandas as pd

# internal
from tools.analysis.plots import common
from tools.analysis.plots.common import LabeledFigure


def build(wide: pd.DataFrame) -> list[LabeledFigure]:
    """Build the system-rollup figures from the system wide frame."""
    candidates = [
        common.categorical_timeline(
            wide, "system.mode", name="system_mode", title="System mode (from SafetyStateMsg)"
        ),
        common.line_panel(
            wide,
            ["system.safe_latched"],
            name="system_safe_latched",
            title="SAFE latch state",
            ylabel="latched (0/1)",
        ),
        common.line_panel(
            wide,
            ["system.total_messages"],
            name="system_total_messages",
            title="Total bus messages per step",
            ylabel="messages / step",
        ),
        common.stacked_counts(
            wide,
            ["system.total_faults", "system.total_commands", "system.total_acks"],
            name="system_event_mix",
            title="Per-step fault / command / ack mix",
        ),
        common.cumulative_lines(
            wide,
            ["system.total_messages", "system.total_faults", "system.total_acks"],
            name="system_cumulative",
            title="Cumulative messages / faults / acks",
        ),
    ]
    return [figure for figure in candidates if figure is not None]
