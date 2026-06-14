"""Downlink figures: queue depth/bytes, AOS gate, per-priority backlog, budget + emission.

Satisfies: REQ-OBS-SIL-001.
"""

from __future__ import annotations

# third-party
import pandas as pd

# internal
from tools.analysis.plots import common
from tools.analysis.plots.common import LabeledFigure

_PRIORITIES = ("FAULT_EVENT", "COMMAND_ACK", "HK_TELEMETRY", "SCIENCE_PRODUCT")


def build(wide: pd.DataFrame) -> list[LabeledFigure]:
    """Build the downlink figures from the downlink wide frame."""
    pending = [f"downlink.pending.{priority}" for priority in _PRIORITIES]
    candidates = [
        common.line_panel(
            wide,
            ["downlink.pending_items", "downlink.next_order"],
            name="downlink_queue",
            title="Downlink queue depth + cumulative enqueued",
            ylabel="count",
        ),
        common.line_panel(
            wide,
            ["downlink.pending_bytes", "downlink.backlog_fraction"],
            name="downlink_backlog",
            title="Queued bytes + backlog vs per-pass budget",
            ylabel="bytes / fraction",
        ),
        common.line_panel(
            wide,
            ["downlink.aos"],
            name="downlink_aos",
            title="AOS gate (link up)",
            ylabel="AOS (0/1)",
        ),
        common.stacked_counts(
            wide,
            pending,
            name="downlink_priority_mix",
            title="Queued downlink items by priority",
            ylabel="queued items",
        ),
        common.line_panel(
            wide,
            ["downlink.item_count"],
            name="downlink_emission",
            title="Downlink items emitted per step",
            ylabel="items / step",
        ),
        common.cumulative_lines(
            wide,
            ["downlink.item_count"],
            name="downlink_emission_cumulative",
            title="Cumulative downlink items emitted",
        ),
    ]
    return [figure for figure in candidates if figure is not None]
