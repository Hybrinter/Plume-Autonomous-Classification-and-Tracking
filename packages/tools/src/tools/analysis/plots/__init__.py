"""Per-app + bus matplotlib figure builders for the SIL telemetry report.

Each group (the ten flight apps, the message bus, and the system rollup) has a builder module
exposing ``build(wide) -> list[LabeledFigure]`` that curates ~4-8 figures from that group's wide
frame using the shared primitives in ``plots.common``. ``PLOT_GROUPS`` collects them (mirroring the
``Signal`` registry's frozen-dataclass-carrying-a-typed-callable convention) so the report can
render and save every group's figures in a stable order. All rendering uses the headless Agg
backend; nothing here drives the SIL.

Contains:
  - GroupPlots / PLOT_GROUPS: the ordered group -> figure-builder registry.
  - build_group_figures: render one group's figures from its wide frame.

Satisfies: REQ-OBS-SIL-001.
"""

from __future__ import annotations

# stdlib
from collections.abc import Callable
from dataclasses import dataclass

# third-party
import pandas as pd

# internal
from tools.analysis.plots import (
    bus,
    command_router,
    downlink,
    electrical,
    fault,
    iss_iface,
    mechanical,
    model_deploy,
    payload,
    storage,
    system,
    thermal,
)
from tools.analysis.plots.common import LabeledFigure

FiguresFn = Callable[[pd.DataFrame], list[LabeledFigure]]


@dataclass(frozen=True, slots=True)
class GroupPlots:
    """A group's figure builder: its group name and the function that renders its figures."""

    group: str
    build: FiguresFn


PLOT_GROUPS: tuple[GroupPlots, ...] = (
    GroupPlots("system", system.build),
    GroupPlots("bus", bus.build),
    GroupPlots("payload", payload.build),
    GroupPlots("fault", fault.build),
    GroupPlots("iss_iface", iss_iface.build),
    GroupPlots("thermal", thermal.build),
    GroupPlots("electrical", electrical.build),
    GroupPlots("command_router", command_router.build),
    GroupPlots("storage", storage.build),
    GroupPlots("downlink", downlink.build),
    GroupPlots("mechanical", mechanical.build),
    GroupPlots("model_deploy", model_deploy.build),
)


def build_group_figures(group: str, wide: pd.DataFrame) -> list[LabeledFigure]:
    """Render the figures for one group from its wide frame.

    Args:
        group: the group name (must be one of PLOT_GROUPS).
        wide: the group's wide frame (step-indexed, with a leading ``t`` column).

    Returns:
        The group's LabeledFigures (possibly empty if no signal had renderable data).

    Raises:
        KeyError: if group is not a registered plot group.
    """
    for entry in PLOT_GROUPS:
        if entry.group == group:
            return entry.build(wide)
    raise KeyError(f"no plot builder for group {group!r}")
