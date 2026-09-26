"""Headless matplotlib figures for inference run directories.

The implementation lives in ``tools.ml_models.analysis.plots``. This module
re-exports the run-directory figures so existing callers keep one import path.

Contains:
  - LabeledFigure: named Figure ready to save.
  - history_figures: train/val loss and metric curves from history.csv.
  - overlay_figures: prediction versus gold overlays from predictions.npz.
  - failure_figures: lowest-scoring preview samples.
  - save_figures: PNG emission.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

from tools.ml_models.analysis.plots import (
    LabeledFigure,
    failure_figures,
    history_figures,
    overlay_figures,
    save_figures,
)

__all__ = [
    "LabeledFigure",
    "failure_figures",
    "history_figures",
    "overlay_figures",
    "save_figures",
]
