"""Model and training-analysis orchestration boundary.

The analysis algorithms are unimplemented; the boundary fails closed with
an explicit unavailable error and creates no output directory.

Contains:
  - analyze_model: the public ``Result`` boundary.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

from pathlib import Path

from flight.libs.types import Err, Result

from tools.ml_models.analysis.config import ModelAnalysisConfig


def analyze_model(cfg: ModelAnalysisConfig) -> Result[Path, str]:
    """Refuse to analyze while model evidence measurement is unimplemented.

    Args:
        cfg: Model analysis inputs.

    Returns:
        Result[Path, str]: Always Err; no analysis bundle is created.
    """
    del cfg
    return Err("model analysis is unavailable until the model evidence phase is implemented")
