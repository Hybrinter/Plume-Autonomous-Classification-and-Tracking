"""Classifier-evidence figure export over the shared model renderer.

Thin family binding: every supplied recipe is drawn verbatim by
``plots.model`` and namespaced under ``figures/classifier/<split>/``.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

from flight.libs.types import Result

from tools.ml_models.analysis.config import PlotConfig
from tools.ml_models.analysis.model_figures import ModelFigure
from tools.ml_models.analysis.plots.dataset import RenderedDatasetFigures
from tools.ml_models.analysis.plots.model import render_model_figures


def render_classifier_figures(
    figures: tuple[ModelFigure, ...],
    cfg: PlotConfig,
) -> Result[RenderedDatasetFigures, str]:
    """Export frozen classifier recipes under the ``classifier`` family."""
    return render_model_figures(figures, cfg, family="classifier")
