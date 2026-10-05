"""Training boundary for GSD-conditioned models over finished datasets.

Standard training is unavailable while the evidence-first training phase is
unimplemented. The public signature is retained so callers and the CLI keep a
stable entry point; every call returns an explicit unavailable error before
any output directory, model, or dataset work begins.

Contains:
  - train: the public ``Result`` boundary.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

from pathlib import Path

from flight.libs.types import Err, Result

from tools.ml_models.train.config import TrainConfig


def train(cfg: TrainConfig | None = None) -> Result[Path, str]:
    """Refuse to run training while the evidence training phase is unimplemented.

    Args:
        cfg: Training configuration. None uses :class:`TrainConfig` defaults.

    Returns:
        Result[Path, str]: Always Err; no run directory, checkpoint, or
        summary is created.
    """
    del cfg
    return Err("standard training is unavailable until the evidence training phase is implemented")
