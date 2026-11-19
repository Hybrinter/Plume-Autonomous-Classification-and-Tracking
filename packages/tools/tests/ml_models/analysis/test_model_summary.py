"""Pure summary identity and completeness contracts over one retained numerical fixture."""

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
from flight.libs.types import Err, Ok
from test_model_measurement import _config, _inputs
from tools.ml_models.analysis.artifacts import artifact_ref
from tools.ml_models.analysis.config import ModelAnalysisConfig
from tools.ml_models.analysis.contracts import ArtifactRef, AvailabilityRecord, CodeIdentity
from tools.ml_models.analysis.model_measurement import ModelMeasurement, measure_model
from tools.ml_models.analysis.model_summary import model_summary


@pytest.fixture
def frozen_summary_inputs(
    tmp_path: Path,
    build_synthetic_dataset: Callable[..., Path],
) -> tuple[
    ModelMeasurement, ModelAnalysisConfig, tuple[ArtifactRef, ...], tuple[AvailabilityRecord, ...]
]:
    dataset = build_synthetic_dataset(tmp_path / "ds", n=9)
    cfg = replace(_config(tmp_path), plot=replace(_config(tmp_path).plot, formats=("png",), dpi=72))
    measured = measure_model(_inputs(dataset, dataset), cfg, tmp_path / "capture")
    assert isinstance(measured, Ok)
    measurement = measured.value
    refs = [
        reference for capture in measurement.captures for reference in capture.evidence.artifacts
    ]
    outputs = [AvailabilityRecord(name="figures", status="AVAILABLE", required=True)]
    charts: list[tuple[str, str, str | None]] = []
    charts.extend(
        (
            "figures/training/" + record.identifier,
            "training_figure:" + record.identifier,
            record.reason,
        )
        for record in measurement.training_figures
    )
    for family, figures in (
        ("classifier", measurement.task_figures),
        ("generalization", measurement.generalization_figures),
    ):
        charts.extend(
            (
                "figures/" + family + "/" + record.identity.split + "/" + record.identifier,
                "model_figure:" + family + ":" + record.identity.split + ":" + record.identifier,
                record.reason,
            )
            for record in figures
        )
    for path, name, reason in charts:
        reference = artifact_ref(
            path + ".png", b"summary-contract-test-artifact", kind="FIGURE", format="png"
        )
        assert isinstance(reference, Ok)
        refs.append(reference.value)
        outputs.append(
            AvailabilityRecord(
                name=name, status="UNAVAILABLE" if reason else "AVAILABLE", reason=reason
            )
        )
    for capture in measurement.captures:
        for gallery in capture.previews.galleries:
            outputs.append(gallery.availability)
            pages = max(1, (len(gallery.rows) + 3) // 4)
            for page in range(pages):
                reference = artifact_ref(
                    "visuals/predictions/" + gallery.identifier + "_" + str(page + 1) + ".png",
                    b"summary-contract-test-artifact",
                    kind="VISUAL",
                    format="png",
                )
                assert isinstance(reference, Ok)
                refs.append(reference.value)
    return measurement, cfg, tuple(refs), tuple(outputs)


def test_model_summary_has_one_category_and_preserves_split_values(
    frozen_summary_inputs: tuple[
        ModelMeasurement,
        ModelAnalysisConfig,
        tuple[ArtifactRef, ...],
        tuple[AvailabilityRecord, ...],
    ],
) -> None:
    measured, cfg, refs, outputs = frozen_summary_inputs
    result = model_summary(
        measured, cfg, CodeIdentity(revision="frozen-test", dirty=False), refs, outputs
    )
    assert isinstance(result, Ok)
    assert result.value.summary_kind == "MODEL_TRAINING_ANALYSIS"
    assert result.value.splits == tuple(capture.evidence for capture in measured.captures)
    assert result.value.training_dataset == measured.training_dataset
    assert result.value.evaluation_dataset == measured.evaluation_dataset
    assert (
        next(
            output for output in result.value.outputs if output.name == "flight_qualification"
        ).status
        == "SKIPPED"
    )


def test_plot_style_or_output_path_does_not_change_measurement_identity(
    frozen_summary_inputs: tuple[
        ModelMeasurement,
        ModelAnalysisConfig,
        tuple[ArtifactRef, ...],
        tuple[AvailabilityRecord, ...],
    ],
) -> None:
    measured, cfg, refs, outputs = frozen_summary_inputs
    code = CodeIdentity(revision="frozen-test", dirty=False)
    original = model_summary(measured, cfg, code, refs, outputs)
    changed = model_summary(
        measured,
        replace(cfg, out="new-output", plot=replace(cfg.plot, font_size=14.0)),
        code,
        refs,
        outputs,
    )
    assert isinstance(original, Ok) and isinstance(changed, Ok)
    assert original.value.measurement_id == changed.value.measurement_id
    assert original.value.splits == changed.value.splits


def test_scoring_and_final_test_settings_cannot_relabel_a_frozen_measurement(
    frozen_summary_inputs: tuple[
        ModelMeasurement,
        ModelAnalysisConfig,
        tuple[ArtifactRef, ...],
        tuple[AvailabilityRecord, ...],
    ],
) -> None:
    measured, cfg, refs, outputs = frozen_summary_inputs
    code = CodeIdentity(revision="frozen-test", dirty=False)
    assert isinstance(
        model_summary(measured, replace(cfg, final_test=True), code, refs, outputs), Err
    )
    assert isinstance(
        model_summary(
            measured,
            replace(cfg, score=replace(cfg.score, classifier_probability_threshold=0.2)),
            code,
            refs,
            outputs,
        ),
        Err,
    )


def test_missing_chart_and_unavailable_relabeling_fail_closed(
    frozen_summary_inputs: tuple[
        ModelMeasurement,
        ModelAnalysisConfig,
        tuple[ArtifactRef, ...],
        tuple[AvailabilityRecord, ...],
    ],
) -> None:
    measured, cfg, refs, outputs = frozen_summary_inputs
    code = CodeIdentity(revision="frozen-test", dirty=False)
    assert isinstance(model_summary(measured, cfg, code, refs[:-1], outputs), Err)
    unavailable = next(
        output
        for output in outputs
        if output.name.startswith("training_figure:") and output.status == "UNAVAILABLE"
    )
    changed = tuple(
        replace(output, status="AVAILABLE", reason=None) if output == unavailable else output
        for output in outputs
    )
    assert isinstance(model_summary(measured, cfg, code, refs, changed), Err)
