"""Fixed-cohort, original-training-baseline and shared capture-budget references."""

from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from flight.libs.types import Err, Ok, Result
from tools.ml_models.analysis import model_measurement as measurement_module
from tools.ml_models.analysis.artifacts import dataset_identity
from tools.ml_models.analysis.capture import CaptureSink
from tools.ml_models.analysis.config import (
    CaptureConfig,
    EvaluationConfig,
    GeneralizationConfig,
    ModelAnalysisConfig,
    ScoreConfig,
)
from tools.ml_models.analysis.contracts import (
    CheckpointIdentity,
    MetricValue,
    SplitEvidence,
)
from tools.ml_models.analysis.evaluate import evaluate_split
from tools.ml_models.analysis.model_measurement import ModelInputs, measure_model
from tools.ml_models.analysis.training import CheckpointRecord, TrainingExecution, TrainingHistory
from tools.ml_models.dataset.build import build_dataset
from tools.ml_models.dataset.manifest import DatasetManifest, load_manifest
from tools.ml_models.dataset.raw import BinSpec, GsdPair, RawTile, RawTileRef
from tools.ml_models.dataset.spec import BuildSpec
from tools.ml_models.train.losses import PlumeLoss, build_loss
from torch import nn


class _TinyModel(nn.Module):
    """Two-input CPU test model with no stochastic layers."""

    def __init__(self, kind: str) -> None:
        super().__init__()
        self.kind = kind
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, image: torch.Tensor, gsd: torch.Tensor) -> torch.Tensor:
        values = image[:, :1] + self.bias
        return values.mean((2, 3)) if self.kind == "classifier" else values


def _inputs(training: Path, evaluation: Path, kind: str = "classifier") -> ModelInputs:
    train_id, evaluation_id = dataset_identity(training), dataset_identity(evaluation)
    assert isinstance(train_id, Ok) and isinstance(evaluation_id, Ok)
    identity = CheckpointIdentity(
        sha256="c" * 64,
        kind="classifier" if kind == "classifier" else "segmentor",
        arch="tiny",
        training_dataset_hash=train_id.value.content_hash,
        epoch=1,
        step=1,
    )
    name = "average_precision" if kind == "classifier" else "foreground_iou_mean_positive_images"
    validation = SplitEvidence(
        task=identity.kind,
        split="val",
        dataset_hash=train_id.value.content_hash,
        dataset_manifest_hash=train_id.value.manifest_hash,
        checkpoint_hash=identity.sha256,
        metrics=(MetricValue(name=name, value=0.5, status="AVAILABLE"),),
    )
    checkpoint = CheckpointRecord(
        path="checkpoints/best.pt",
        identity=identity,
        metric=name,
        direction="MAXIMIZE",
        value=0.5,
        validation=validation,
    )
    execution = TrainingExecution(
        kind=identity.kind,
        arch="tiny",
        dataset_hash=train_id.value.content_hash,
        conditioning="film-log-gsd-v1",
        config={},
        provenance={},
        metric=name,
        epoch=1,
        optimizer_step=1,
        samples_seen=1,
        best=checkpoint,
        selected=checkpoint,
        status="INTERRUPTED",
        stop_reason="partial fixture",
    )
    return ModelInputs(
        TrainingHistory(execution, (), ("Captured prefix only",)),
        identity,
        train_id.value,
        evaluation_id.value,
        training,
        evaluation,
        load_manifest(training / "dataset.json"),
        load_manifest(evaluation / "dataset.json"),
        _TinyModel(kind),
        build_loss("bce"),
        (),
    )


def _config(tmp_path: Path, final_test: bool = False) -> ModelAnalysisConfig:
    return ModelAnalysisConfig(
        run="fixture-run",
        out=str(tmp_path / "published"),
        final_test=final_test,
        capture=CaptureConfig(max_preview_images=2, examples_per_family=2),
        score=ScoreConfig(
            pixel_histogram_bins=8,
            n_calibration_bins=4,
            probability_thresholds=(0.0, 0.5, 1.0),
            min_blob_area_px=1,
        ),
        generalization=GeneralizationConfig(bootstrap_replicates=4),
    )


def _spy(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[tuple[Path, str, float]],
) -> None:
    def measured(
        model: nn.Module,
        dataset: Path,
        manifest: DatasetManifest,
        cfg: EvaluationConfig,
        objective: PlumeLoss | None = None,
        capture: CaptureSink | None = None,
    ) -> Result[SplitEvidence, str]:
        calls.append((dataset, cfg.split, cfg.score.classifier_probability_threshold))
        return evaluate_split(model, dataset, manifest, cfg, objective, capture)

    monkeypatch.setattr(measurement_module, "evaluate_split", measured)


@pytest.mark.parametrize("kind", ("classifier", "segmentor"))
def test_default_analysis_never_evaluates_test_and_keeps_one_selected_identity(
    tmp_path: Path,
    build_synthetic_dataset: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    dataset = build_synthetic_dataset(tmp_path / "ds", n=9)
    inputs = _inputs(dataset, dataset, kind)
    calls: list[tuple[Path, str, float]] = []
    _spy(monkeypatch, calls)
    result = measure_model(inputs, _config(tmp_path), tmp_path / "capture")
    assert isinstance(result, Ok)
    assert [(root, split) for root, split, _ in calls] == [(dataset, "train"), (dataset, "val")]
    assert [capture.evidence.split for capture in result.value.captures] == ["train", "val"]
    assert all(
        capture.evidence.checkpoint_hash == inputs.checkpoint.sha256
        for capture in result.value.captures
    )
    assert result.value.baseline.training_dataset_hash == inputs.training_dataset.content_hash
    assert result.value.development is not None
    assert (
        next(output for output in result.value.outputs if output.name == "final_test").status
        == "SKIPPED"
    )
    assert not any(
        series.name == "selected-checkpoint test"
        for figure in result.value.training_figures
        for series in figure.series
    )


def test_final_test_is_explicit_one_point_at_the_selected_step_without_threshold_fitting(
    tmp_path: Path,
    build_synthetic_dataset: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = build_synthetic_dataset(tmp_path / "ds", n=9)
    inputs = _inputs(dataset, dataset)
    cfg = _config(tmp_path, True)
    cfg = replace(cfg, score=replace(cfg.score, classifier_probability_threshold=0.3))
    calls: list[tuple[Path, str, float]] = []
    _spy(monkeypatch, calls)
    result = measure_model(inputs, cfg, tmp_path / "capture")
    assert isinstance(result, Ok)
    assert [split for _, split, _ in calls] == ["train", "val", "test"]
    assert all(threshold == 0.3 for _, _, threshold in calls)
    figure = next(
        figure
        for figure in result.value.training_figures
        if figure.identifier == "evaluation_loss_linear"
    )
    point = next(series for series in figure.series if series.name == "selected-checkpoint test")
    assert point.x == (1.0,) and len(point.y) == 1 and point.points_only
    assert result.value.checkpoint == inputs.checkpoint


def test_external_dataset_is_evaluation_only_and_baseline_uses_original_training_rows(
    tmp_path: Path,
    build_synthetic_dataset: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = build_synthetic_dataset(tmp_path / "original", n=9)
    external = build_synthetic_dataset(tmp_path / "external", n=12)
    inputs = _inputs(original, external)
    calls: list[tuple[Path, str, float]] = []
    _spy(monkeypatch, calls)
    result = measure_model(inputs, _config(tmp_path), tmp_path / "capture")
    assert isinstance(result, Ok)
    assert [(root, split) for root, split, _ in calls] == [
        (original, "train"),
        (external, "train"),
        (external, "val"),
    ]
    assert result.value.baseline_capture is not None
    original_rows = result.value.baseline_capture.rows
    assert result.value.baseline.n_images == len(original_rows)
    assert result.value.baseline.training_prevalence == sum(
        row.label for row in original_rows
    ) / len(original_rows)
    assert result.value.development is None
    assert all(
        capture.evidence.dataset_hash == inputs.evaluation_dataset.content_hash
        for capture in result.value.captures
    )
    assert next(
        output for output in result.value.outputs if output.name == "development_gaps"
    ).reason


def test_all_persisted_previews_share_one_execution_budget(
    tmp_path: Path,
    build_synthetic_dataset: Callable[..., Path],
) -> None:
    dataset = build_synthetic_dataset(tmp_path / "ds", n=9)
    cfg = _config(tmp_path, True)
    result = measure_model(_inputs(dataset, dataset), cfg, tmp_path / "capture")
    assert isinstance(result, Ok)
    files = tuple(file for capture in result.value.captures for file in capture.files)
    assert sum("/previews/" in file.path and file.path.endswith(".npz") for file in files) <= 2
    assert sum(len(file.data) for file in files) <= cfg.capture.max_capture_bytes
    assert not Path(cfg.out).exists()
    val = next(capture for capture in result.value.captures if capture.evidence.split == "val")
    representatives = next(
        gallery for gallery in val.previews.galleries if gallery.family == "representative"
    )
    assert representatives.rows
    assert representatives.availability.status == "AVAILABLE"
    assert not val.previews.previews


def test_budget_failure_returns_err_before_any_publication(
    tmp_path: Path,
    build_synthetic_dataset: Callable[..., Path],
) -> None:
    dataset = build_synthetic_dataset(tmp_path / "ds", n=9)
    cfg = _config(tmp_path)
    cfg = replace(cfg, capture=replace(cfg.capture, max_capture_bytes=1))
    result = measure_model(_inputs(dataset, dataset), cfg, tmp_path / "capture")
    assert isinstance(result, Err)
    assert not Path(cfg.out).exists()


def test_mismatched_checkpoint_or_dataset_snapshot_fails_before_evaluation(
    tmp_path: Path,
    build_synthetic_dataset: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = build_synthetic_dataset(tmp_path / "ds", n=9)
    inputs = _inputs(dataset, dataset)
    calls: list[tuple[Path, str, float]] = []
    _spy(monkeypatch, calls)
    unknown = replace(inputs.checkpoint, sha256="f" * 64)
    assert isinstance(
        measure_model(
            replace(inputs, checkpoint=unknown), _config(tmp_path), tmp_path / "bad_checkpoint"
        ),
        Err,
    )
    changed = replace(inputs.evaluation_dataset, content_hash="f" * 64)
    assert isinstance(
        measure_model(
            replace(inputs, evaluation_dataset=changed), _config(tmp_path), tmp_path / "bad_dataset"
        ),
        Err,
    )
    assert isinstance(
        measure_model(
            inputs, replace(_config(tmp_path), checkpoint="last"), tmp_path / "unavailable_last"
        ),
        Err,
    )
    assert isinstance(
        measure_model(
            inputs, replace(_config(tmp_path), checkpoint="../other"), tmp_path / "unsafe_selector"
        ),
        Err,
    )
    assert calls == []


def test_external_test_only_dataset_requires_explicit_exposure(
    tmp_path: Path,
    build_synthetic_dataset: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = build_synthetic_dataset(tmp_path / "original", n=9)

    class _TestOnlyMaskSource:
        name = "test-only-mask-source"
        band_names: tuple[str, ...] = ("BLUE", "GREEN", "RED")
        domain = "unit"
        source_ref = ""
        bins: tuple[BinSpec, ...] = ()

        def index(self) -> tuple[RawTileRef, ...]:
            return tuple(
                RawTileRef(
                    tile_id=str(index),
                    group_id="g" + str(index // 2),
                    label=float(index % 2),
                    has_mask=index // 2 == 3,
                    gsd=GsdPair(2.0, 3.0),
                    height=2,
                    width=3,
                    frame_id=None,
                    grid_rc=None,
                    bin_id="",
                )
                for index in range(8)
            )

        def iter_tiles(self) -> Iterator[RawTile]:
            for ref in self.index():
                mask = np.full((1, 2, 3), int(ref.label), dtype=np.uint8) if ref.has_mask else None
                yield RawTile(ref, np.full((3, 2, 3), ref.label, dtype=np.float32), mask)

    external = tmp_path / "external-test-only"
    build_dataset(_TestOnlyMaskSource(), external, BuildSpec(tasks=("segmentor",)))
    inputs = _inputs(original, external, "segmentor")
    assert {shard.split for shard in inputs.evaluation_manifest.shards} == {"test"}
    calls: list[tuple[Path, str, float]] = []
    _spy(monkeypatch, calls)
    refused = measure_model(inputs, _config(tmp_path), tmp_path / "not_final")
    assert isinstance(refused, Err)
    assert [split for root, split, _ in calls if root == external] == []
    calls.clear()
    final = measure_model(inputs, _config(tmp_path, True), tmp_path / "final")
    assert isinstance(final, Ok)
    assert [(root, split) for root, split, _ in calls] == [(original, "train"), (external, "test")]
    assert [capture.evidence.split for capture in final.value.captures] == ["test"]
    assert final.value.development is None
    assert {output.name for output in final.value.outputs if output.status == "UNAVAILABLE"} >= {
        "evaluation:train",
        "evaluation:val",
        "development_gaps",
    }
