"""Command-router figures: armed hazardous commands, SAFE mirror, routing throughput.

Satisfies: REQ-OBS-SIL-001.
"""

from __future__ import annotations

# third-party
import pandas as pd

# internal
from tools.analysis.plots import common
from tools.analysis.plots.common import LabeledFigure


def build(wide: pd.DataFrame) -> list[LabeledFigure]:
    """Build the command-router figures from the command_router wide frame."""
    candidates = [
        common.line_panel(
            wide,
            ["command_router.armed", "command_router.safe_latched"],
            name="router_state",
            title="Armed hazardous commands + SAFE mirror",
            ylabel="count",
        ),
        common.stacked_counts(
            wide,
            [
                "command_router.command_count",
                "command_router.routed_count",
                "command_router.ack_count",
                "command_router.unroutable_count",
            ],
            name="router_throughput",
            title="Per-step command routing throughput",
        ),
        common.cumulative_lines(
            wide,
            ["command_router.routed_count", "command_router.ack_count"],
            name="router_cumulative",
            title="Cumulative routed commands + acks",
        ),
    ]
    return [figure for figure in candidates if figure is not None]
