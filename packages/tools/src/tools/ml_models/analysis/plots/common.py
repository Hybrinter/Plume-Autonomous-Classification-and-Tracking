"""Figure/export conventions and the render boundary.

Rendering is unimplemented; the boundary fails closed with an explicit
unavailable error and creates no output directory.

Contains:
  - render_analysis: the public ``Result`` boundary.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

from pathlib import Path

from flight.libs.types import Err, Result

from tools.ml_models.analysis.config import PlotConfig


def render_analysis(evidence_dir: Path, cfg: PlotConfig, out: Path) -> Result[Path, str]:
    """Refuse to render while figure generation is unimplemented.

    Args:
        evidence_dir: Frozen evidence directory.
        cfg: Figure output settings.
        out: Destination figure directory.

    Returns:
        Result[Path, str]: Always Err; no figures are created.
    """
    del evidence_dir, cfg, out
    return Err("figure rendering is unavailable until the plotting phase is implemented")
