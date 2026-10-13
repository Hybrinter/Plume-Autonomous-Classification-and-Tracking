"""One exhaustive split-evaluation boundary.

The evaluation algorithm is unimplemented; the boundary fails closed with
an explicit unavailable error and produces no evidence.

Contains:
  - evaluate_split: the public ``Result`` boundary.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

from pathlib import Path

from flight.libs.types import Err, Result
from torch import nn

from tools.ml_models.analysis.capture import CaptureSink
from tools.ml_models.analysis.config import EvaluationConfig
from tools.ml_models.analysis.contracts import SplitEvidence
from tools.ml_models.dataset.manifest import DatasetManifest
from tools.ml_models.train.losses import PlumeLoss


def evaluate_split(
    model: nn.Module,
    dataset: Path,
    manifest: DatasetManifest,
    cfg: EvaluationConfig,
    objective: PlumeLoss | None = None,
    capture: CaptureSink | None = None,
) -> Result[SplitEvidence, str]:
    """Refuse to evaluate while the shared evaluator is unimplemented.

    Args:
        model: Conditioned model called as ``model(image, encoded_gsd)``.
        dataset: Finished dataset directory.
        manifest: Parsed dataset manifest.
        cfg: Evaluation inputs.
        objective: Optional configured objective for loss records.
        capture: Optional sink for captured predictions.

    Returns:
        Result[SplitEvidence, str]: Always Err; no inference runs and no
        outputs are created.
    """
    del model, dataset, manifest, cfg, objective, capture
    return Err("split evaluation is unavailable until the evidence evaluation phase is implemented")
