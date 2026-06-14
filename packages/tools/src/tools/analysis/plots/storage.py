"""Storage figures: live byte usage + headroom, entries, eviction, fullness, write flow.

Satisfies: REQ-OBS-SIL-001.
"""

from __future__ import annotations

# third-party
import pandas as pd

# internal
from tools.analysis.plots import common
from tools.analysis.plots.common import LabeledFigure


def build(wide: pd.DataFrame) -> list[LabeledFigure]:
    """Build the storage figures from the storage wide frame."""
    candidates = [
        common.line_panel(
            wide,
            ["storage.total_bytes", "storage.headroom_bytes"],
            name="storage_bytes",
            title="Live stored bytes + quota headroom",
            ylabel="bytes",
        ),
        common.line_panel(
            wide,
            ["storage.entries", "storage.next_order", "storage.dropped_count"],
            name="storage_entries",
            title="Live entries + cumulative stored/evicted",
            ylabel="count",
        ),
        common.line_panel(
            wide,
            ["storage.fraction_full"],
            name="storage_fullness",
            title="Storage quota fraction used",
            ylabel="fraction",
        ),
        common.stacked_counts(
            wide,
            ["storage.write_count", "storage.telemetry_persisted", "storage.full_fault_count"],
            name="storage_activity",
            title="Per-step storage write + persist + full-fault activity",
        ),
        common.cumulative_lines(
            wide,
            ["storage.write_count"],
            name="storage_write_cumulative",
            title="Cumulative storage writes",
        ),
    ]
    return [figure for figure in candidates if figure is not None]
