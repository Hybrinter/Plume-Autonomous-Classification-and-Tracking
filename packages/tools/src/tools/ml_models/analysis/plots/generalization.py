"""Generalization-evidence figure export over the shared model renderer.

Thin family binding: every supplied recipe is drawn verbatim by
``plots.model`` and namespaced under ``figures/generalization/<split>/``.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

from flight.libs.types import Result

from tools.ml_models.analysis.config import PlotConfig
from tools.ml_models.analysis.model_figures import ModelFigure
from tools.ml_models.analysis.plots.dataset import RenderedDatasetFigures
from tools.ml_models.analysis.plots.model import render_model_figures


def render_generalization_figures(
    figures: tuple[ModelFigure, ...],
    cfg: PlotConfig,
) -> Result[RenderedDatasetFigures, str]:
    """Export frozen generalization recipes under the ``generalization`` family."""
    return render_model_figures(figures, cfg, family="generalization")
