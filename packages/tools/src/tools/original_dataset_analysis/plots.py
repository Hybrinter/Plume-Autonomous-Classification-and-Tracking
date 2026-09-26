"""Bar charts for the band matrix.

The implementation lives in ``tools.ml_models.analysis.plots``. This module
re-exports the study writers.

Contains:
  - write_band_bars: one bar per subset name.
  - write_metric_bars: one chart for one metric.
  - write_delta_bars: score minus the 12-band baseline.
  - write_loss_curve: linear-epoch loss with a labeled selected epoch.
  - write_pr_curve: precision-recall curve.
  - write_gsd_lines: ``pr_auc`` and ``native_dice`` against ground-sample distance.
"""

from __future__ import annotations

from tools.ml_models.analysis.plots import (
    write_band_bars,
    write_delta_bars,
    write_gsd_lines,
    write_loss_curve,
    write_metric_bars,
    write_pr_curve,
)

__all__ = [
    "write_band_bars",
    "write_delta_bars",
    "write_gsd_lines",
    "write_loss_curve",
    "write_metric_bars",
    "write_pr_curve",
]
