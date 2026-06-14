"""Thermal figures: temperature versus the (effective) limit, override, and fault activity.

Satisfies: REQ-OBS-SIL-001.
"""

from __future__ import annotations

# third-party
import pandas as pd

# internal
from tools.analysis.plots import common
from tools.analysis.plots.common import LabeledFigure


def build(wide: pd.DataFrame) -> list[LabeledFigure]:
    """Build the thermal figures from the thermal wide frame."""
    candidates = [
        common.value_with_limit(
            wide,
            "thermal.temperature_c",
            "thermal.limit_c",
            name="thermal_temperature",
            title="Temperature vs effective limit",
            ylabel="degC",
        ),
        common.line_panel(
            wide,
            ["thermal.limit_override_c"],
            name="thermal_limit_override",
            title="Commanded thermal-limit override",
            ylabel="degC",
        ),
        common.stacked_counts(
            wide,
            ["thermal.sample_count", "thermal.fault_count"],
            name="thermal_activity",
            title="Per-step thermal samples + over-limit faults",
        ),
        common.cumulative_lines(
            wide,
            ["thermal.fault_count"],
            name="thermal_fault_cumulative",
            title="Cumulative thermal over-limit faults",
        ),
    ]
    return [figure for figure in candidates if figure is not None]
