"""Model and training-analysis orchestration boundary.

``analyze_model`` verifies every frozen source input, measures the selected
checkpoint once inside a private capture workspace, assembles the frozen
evidence bundle, renders all recipes, binds one canonical summary and
publishes exclusively. A failure before publication leaves no output
behind; a failure during publication leaves the reserved directory with an
``.incomplete`` marker so it is never mistaken for a finished bundle.

Contains:
  - analyze_model: the public ``Result`` boundary.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path

from flight.libs.types import Err, Result

from tools.ml_models.analysis.artifacts import (
    BundleFile,
    artifact_ref,
    publish_bundle,
)
from tools.ml_models.analysis.config import ModelAnalysisConfig
from tools.ml_models.analysis.dataset_artifacts import code_identity
from tools.ml_models.analysis.model_artifacts import (
    assemble_model_artifacts,
    combined_prediction_capture,
)
from tools.ml_models.analysis.model_inputs import load_model_inputs
from tools.ml_models.analysis.model_measurement import measure_model
from tools.ml_models.analysis.model_render import (
    render_model_outputs,
    rendering_document,
)
from tools.ml_models.analysis.model_summary import model_summary


def analyze_model(cfg: ModelAnalysisConfig) -> Result[Path, str]:
    """Measure, freeze, render and exclusively publish one model-analysis bundle.

    All input verification happens before any inference; ``measure_model``
    runs once inside a temporary capture workspace. Codecs and renderers
    must all succeed before ``publish_bundle`` reserves the output, which
    never overwrites an existing directory and never sits inside the run or
    either dataset root.
    """
    inputs = load_model_inputs(cfg)
    if isinstance(inputs, Err):
        return inputs
    out = Path(cfg.out)
    try:
        with tempfile.TemporaryDirectory(prefix=".model-capture-") as workspace:
            measured = measure_model(inputs.value, cfg, Path(workspace))
            if isinstance(measured, Err):
                return measured
            assembled = assemble_model_artifacts(measured.value, cfg)
            if isinstance(assembled, Err):
                return assembled
            rendered = render_model_outputs(
                measured.value.training_figures,
                measured.value.task_figures,
                measured.value.generalization_figures,
                combined_prediction_capture(measured.value),
                measured.value.checkpoint.kind,
                cfg.plot,
            )
            if isinstance(rendered, Err):
                return rendered
            refs = assembled.value.refs + rendered.value.refs
            files = assembled.value.files + rendered.value.files
            summary = model_summary(
                measured.value,
                cfg,
                code_identity(),
                refs,
                rendered.value.outputs,
            )
            if isinstance(summary, Err):
                return summary
            document = rendering_document(summary.value.measurement_id, cfg.plot)
            if isinstance(document, Err):
                return document
            rendering_ref = artifact_ref(
                "rendering.json", document.value, kind="REFERENCE", format="json"
            )
            if isinstance(rendering_ref, Err):
                return rendering_ref
            published = replace(summary.value, artifacts=refs + (rendering_ref.value,))
            return publish_bundle(
                out,
                published,
                files + (BundleFile("rendering.json", document.value),),
                dataset_root=inputs.value.evaluation_root,
            )
    except (OSError, ValueError, TypeError, RuntimeError, OverflowError) as exc:
        return Err(f"model analysis failed before publication: {exc}")
