"""ISS-interface figures: link state, TM sequence, ingress + upload activity, station traffic.

Satisfies: REQ-OBS-SIL-001.
"""

from __future__ import annotations

# third-party
import pandas as pd

# internal
from tools.analysis.plots import common
from tools.analysis.plots.common import LabeledFigure


def build(wide: pd.DataFrame) -> list[LabeledFigure]:
    """Build the ISS-interface figures from the iss_iface wide frame."""
    candidates = [
        common.categorical_timeline(
            wide, "iss_iface.link_state", name="iss_link_state", title="Station link state"
        ),
        common.line_panel(
            wide,
            ["iss_iface.tm_sequence", "iss_iface.station_sent_total"],
            name="iss_downlink_traffic",
            title="Outbound TM sequence + station packets sent",
            ylabel="count (cumulative)",
        ),
        common.line_panel(
            wide,
            [
                "iss_iface.known_sources",
                "iss_iface.upload_chunks_buffered",
                "iss_iface.upload_total_chunks",
                "iss_iface.upload_progress",
            ],
            name="iss_uplink_state",
            title="Ingress replay guard + upload reassembly",
            ylabel="count / fraction",
        ),
        common.stacked_counts(
            wide,
            [
                "iss_iface.command_published",
                "iss_iface.ack_count",
                "iss_iface.link_state_count",
                "iss_iface.model_staged_count",
                "iss_iface.upload_chunk_count",
            ],
            name="iss_message_flow",
            title="Per-step ISS message flow",
        ),
        common.cumulative_lines(
            wide,
            ["iss_iface.command_published", "iss_iface.ack_count"],
            name="iss_cumulative",
            title="Cumulative commands published + acks",
        ),
    ]
    return [figure for figure in candidates if figure is not None]
