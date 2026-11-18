"""Split scoring and late-fusion weight search for trained runs.

Contains:
  - LateFusionResult: chosen VIS weight, validation scores, and test metrics.
  - load_run: verified model inputs for one run on one dataset.
  - score_split: aggregate metrics of one model on one split.
  - check_gsd_reference: require one GSD reference for both branches and the dataset.
  - late_fusion: search the VIS weight on validation and score test once.

Scoring reuses the model analysis evaluator without capture, so the metrics
match a full analysis but cost one inference pass. The analysis
configuration supplies the batch size, device, and score settings; its
``run`` and ``out`` values are replaced. ``out`` is never written.

Satisfies: REQ-AIML-COMP-001.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path

from flight.libs.types import Err, Ok, Result
from torch import nn

from tools.ml_models.analysis.config import EvaluationConfig, ModelAnalysisConfig
from tools.ml_models.analysis.contracts import Split
from tools.ml_models.analysis.evaluate import evaluate_split
from tools.ml_models.analysis.model_inputs import load_model_inputs
from tools.ml_models.analysis.model_measurement import ModelInputs
from tools.ml_models.fusion.late import LateFusion, select_weight


@dataclass(frozen=True, slots=True)
class LateFusionResult:
    """Outcome of one late-fusion weight search.

    Attributes:
        vis_weight: VIS weight chosen on validation.
        val_by_weight: Validation score per searched VIS weight.
        test: Test metrics at the chosen weight.
    """

    vis_weight: float
    val_by_weight: dict[float, float]
    test: dict[str, float]


def load_run(
    base: ModelAnalysisConfig, run: Path, dataset: Path | None
) -> Result[ModelInputs, str]:
    """Load and verify one run's selected checkpoint and evaluation dataset.

    Args:
        base: Analysis settings; ``run`` and ``out`` are replaced. ``out`` is a
            never-written sibling of ``run``; a path inside the run is rejected.
        run: Run directory.
        dataset: Evaluation dataset; ``None`` uses the training dataset.

    Returns:
        Result[ModelInputs, str]: Verified inputs, or the load error.
    """
    cfg = replace(
        base,
        run=str(run),
        out=str(run.parent / f"{run.name}_fusion_unused"),
        dataset=None if dataset is None else str(dataset),
    )
    return load_model_inputs(cfg)


def score_split(
    model: nn.Module, inputs: ModelInputs, base: ModelAnalysisConfig, split: Split
) -> Result[dict[str, float], str]:
    """Return the aggregate metrics of ``model`` on one split of the inputs' dataset.

    Args:
        model: Model called as ``model(image, gsd)``.
        inputs: Verified inputs that name the evaluation dataset and task.
        base: Analysis settings for batch size, device, and scoring.
        split: ``train``, ``val``, or ``test``.

    Returns:
        Result[dict[str, float], str]: Metric name to value for every finite
        scalar metric, or the evaluation error.
    """
    try:
        cfg = EvaluationConfig(
            kind=inputs.checkpoint.kind,
            split=split,
            batch_size=base.batch_size,
            device=base.device,
            score=base.score,
        )
    except ValueError as exc:
        return Err(f"bad evaluation settings: {exc}")
    evidence = evaluate_split(model, inputs.evaluation_root, inputs.evaluation_manifest, cfg)
    if isinstance(evidence, Err):
        return evidence
    return Ok({m.name: m.value for m in evidence.value.metrics if isinstance(m.value, float)})


def check_gsd_reference(vis_m: float, swir_m: float, paired_m: float) -> Result[None, str]:
    """Require the two branch training references to equal the evaluation reference.

    Args:
        vis_m: GSD reference of the VIS run's training dataset in m.
        swir_m: GSD reference of the SWIR run's training dataset in m.
        paired_m: GSD reference of the paired evaluation dataset in m.

    Returns:
        Result[None, str]: ``Ok(None)`` when all three agree, else the mismatch.

    Notes:
        The model GSD input is ``ln(gsd / reference)``. The evaluator encodes
        it with the paired reference and late fusion passes the same tensor to
        both branches, so a branch trained on another reference sees a shifted
        input.
    """
    for name, value in (("VIS", vis_m), ("SWIR", swir_m)):
        if not math.isclose(value, paired_m, rel_tol=1e-9):
            return Err(
                f"{name} run GSD reference {value} m differs from the paired "
                f"dataset reference {paired_m} m"
            )
    return Ok(None)


def late_fusion(
    vis: ModelInputs,
    swir: ModelInputs,
    paired: ModelInputs,
    base: ModelAnalysisConfig,
    metric: str,
    weights: tuple[float, ...],
) -> Result[LateFusionResult, str]:
    """Choose the VIS weight on validation, then score test once at that weight.

    Args:
        vis: VIS-only run inputs; only its model is used.
        swir: SWIR-only run inputs; only its model is used.
        paired: Inputs that name the paired evaluation dataset and the task.
        base: Analysis settings for batch size, device, and scoring.
        metric: Validation metric to maximize, such as ``average_precision``.
        weights: VIS weights to search.

    Returns:
        Result[LateFusionResult, str]: The search outcome, or the first error.
    """
    reference = check_gsd_reference(
        vis.training_manifest.gsd_reference_m,
        swir.training_manifest.gsd_reference_m,
        paired.evaluation_manifest.gsd_reference_m,
    )
    if isinstance(reference, Err):
        return reference
    model = LateFusion(vis.model, swir.model)
    val_by_weight: dict[float, float] = {}
    for weight in weights:
        model.set_vis_weight(weight)
        scored = score_split(model, paired, base, "val")
        if isinstance(scored, Err):
            return scored
        if metric not in scored.value:
            return Err(f"validation metric {metric} is not available")
        val_by_weight[weight] = scored.value[metric]
    best = select_weight(val_by_weight)
    model.set_vis_weight(best)
    test = score_split(model, paired, base, "test")
    if isinstance(test, Err):
        return test
    return Ok(LateFusionResult(vis_weight=best, val_by_weight=val_by_weight, test=test.value))
