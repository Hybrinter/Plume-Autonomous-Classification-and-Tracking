"""Message-bus figures: publish throughput, queue depth, drops/overflow, and the type mix.

Satisfies: REQ-OBS-SIL-001.
"""

from __future__ import annotations

# third-party
import pandas as pd

# internal
from tools.analysis.plots import common
from tools.analysis.plots.common import LabeledFigure

# A representative spread of message types for the per-type stacked/throughput views.
_KEY_TYPES = (
    "InferenceResult",
    "TelemetryEvent",
    "FaultEvent",
    "GimbalCommand",
    "Heartbeat",
    "SafetyState",
    "LinkState",
    "DownlinkItem",
    "Command",
    "CommandAck",
)


def build(wide: pd.DataFrame) -> list[LabeledFigure]:
    """Build the message-bus figures from the bus wide frame."""
    published = [f"bus.published.{name}" for name in _KEY_TYPES]
    depths = [f"bus.depth.{name}" for name in _KEY_TYPES]
    candidates = [
        common.line_panel(
            wide,
            ["bus.published.total"],
            name="bus_published_total",
            title="All messages published per step",
            ylabel="messages / step",
        ),
        common.stacked_counts(
            wide,
            published,
            name="bus_published_mix",
            title="Per-step publish mix (key message types)",
        ),
        common.line_panel(
            wide,
            ["bus.depth.total", *depths],
            name="bus_queue_depth",
            title="Bus consumer backlog (queue depth)",
            ylabel="queued messages",
        ),
        common.line_panel(
            wide,
            ["bus.dropped.total", "bus.overflow.total"],
            name="bus_loss",
            title="Cumulative drops + soft-bound overflow",
            ylabel="count (cumulative)",
        ),
        common.line_panel(
            wide,
            ["bus.types_active"],
            name="bus_types_active",
            title="Distinct message types active per step",
            ylabel="types",
        ),
        common.cumulative_lines(
            wide,
            ["bus.published.total"],
            name="bus_published_cumulative",
            title="Cumulative messages published",
        ),
    ]
    return [figure for figure in candidates if figure is not None]
