"""Dataset measurement orchestration boundary.

The measurement algorithms are unimplemented; the boundary fails closed
with an explicit unavailable error and creates no output directory.

Contains:
  - analyze_dataset: the public ``Result`` boundary.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

from pathlib import Path

from flight.libs.types import Err, Result

from tools.ml_models.analysis.config import DatasetAnalysisConfig


def analyze_dataset(cfg: DatasetAnalysisConfig) -> Result[Path, str]:
    """Refuse to analyze while dataset measurement is unimplemented.

    Args:
        cfg: Dataset analysis inputs.

    Returns:
        Result[Path, str]: Always Err; no analysis bundle is created.
    """
    del cfg
    return Err("dataset analysis is unavailable until the dataset evidence phase is implemented")
