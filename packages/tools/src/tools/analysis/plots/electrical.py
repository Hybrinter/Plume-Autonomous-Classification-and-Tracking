"""Electrical figures: bus power versus the limit and over-limit fault activity.

Satisfies: REQ-OBS-SIL-001.
"""

from __future__ import annotations

# third-party
import pandas as pd

# internal
from tools.analysis.plots import common
from tools.analysis.plots.common import LabeledFigure


def build(wide: pd.DataFrame) -> list[LabeledFigure]:
    """Build the electrical figures from the electrical wide frame."""
    candidates = [
        common.value_with_limit(
            wide,
            "electrical.power_w",
            "electrical.limit_w",
            name="electrical_power",
            title="Bus power vs limit",
            ylabel="W",
        ),
        common.stacked_counts(
            wide,
            ["electrical.sample_count", "electrical.fault_count"],
            name="electrical_activity",
            title="Per-step electrical samples + over-limit faults",
        ),
        common.cumulative_lines(
            wide,
            ["electrical.fault_count"],
            name="electrical_fault_cumulative",
            title="Cumulative power over-limit faults",
        ),
    ]
    return [figure for figure in candidates if figure is not None]
