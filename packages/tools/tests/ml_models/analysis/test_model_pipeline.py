"""End-to-end checks for the model measure-freeze-render-publish pipeline."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import tools.ml_models.analysis.evaluate as evaluate_mod
import tools.ml_models.analysis.model_measurement as measure_mod
import torch
from flight.libs.types import Err, Ok, Result
from tools.ml_models.analysis.artifacts import (
    INCOMPLETE_FILENAME,
    decode_summary,
    encode_summary,
    verify_bundle,
)
from tools.ml_models.analysis.capture import CaptureSink
from tools.ml_models.analysis.config import (
    CaptureConfig,
    DatasetAnalysisConfig,
    EvaluationConfig,
    GeneralizationConfig,
    ModelAnalysisConfig,
    PlotConfig,
)
from tools.ml_models.analysis.contracts import (
    MetricSupport,
    MetricValue,
    SplitEvidence,
)
from tools.ml_models.analysis.dataset import analyze_dataset
from tools.ml_models.analysis.model import analyze_model
from tools.ml_models.analysis.plots.common import render_analysis
from tools.ml_models.analysis.summaries import ModelTrainingSummary
from tools.ml_models.analysis.training import read_training_history
from tools.ml_models.dataset.augment import AugmentRecipe
from tools.ml_models.dataset.build import build_dataset
from tools.ml_models.dataset.manifest import DatasetManifest
from tools.ml_models.dataset.raw import BinSpec, GsdPair, RawTile, RawTileRef
from tools.ml_models.dataset.spec import BuildSpec
from tools.ml_models.train.config import TrainConfig
from tools.ml_models.train.loop import train
from tools.ml_models.train.losses import PlumeLoss

_TILE_HW = (8, 10)
_BANDS: tuple[str, ...] = ("pan", "red", "nir")
_GSD_WINDOW = (
    GsdPair(15.0, 15.0),
    GsdPair(18.0, 21.0),
    GsdPair(25.0, 30.0),
)


class _TinySource:
    """Planted-blob ``RawSource`` on tiny 8x10 tiles; every group has both classes."""

    name = "tiny"
    band_names: tuple[str, ...] = _BANDS
    domain = "unit"
    source_ref = ""
    bins: tuple[BinSpec, ...] = ()

    def __init__(self, n: int = 12, seed: int = 0) -> None:
        if n < 3:
            raise ValueError(f"fixture source needs at least 3 tiles; got {n}")
        self._tiles = _generate(n, seed)

    def index(self) -> tuple[RawTileRef, ...]:
        return tuple(tile.ref for tile in self._tiles)

    def iter_tiles(self) -> Iterator[RawTile]:
        yield from self._tiles


def _generate(n: int, seed: int) -> tuple[RawTile, ...]:
    height, width = _TILE_HW
    generator = np.random.default_rng(seed)
    tiles: list[RawTile] = []
    for index in range(n):
        label = float(index % 2)
        group = f"g{index // 3}"
        image = generator.random((3, height, width), dtype=np.float32) * np.float32(0.2)
        mask = np.zeros((1, height, width), dtype=np.uint8)
        if label:
            image[:, 2:6, 3:7] = np.float32(0.9)
            mask[0, 2:6, 3:7] = np.uint8(1)
        tiles.append(
            RawTile(
                ref=RawTileRef(
                    tile_id=f"t{index:03d}",
                    group_id=group,
                    label=label,
                    has_mask=True,
                    gsd=_GSD_WINDOW[index % len(_GSD_WINDOW)],
                    height=height,
                    width=width,
                    frame_id=group,
                    grid_rc=(index // 5, index % 5),
                    bin_id="",
                ),
                image=image,
                mask=mask,
            )
        )
    return tuple(tiles)


def _dataset(dest: Path, seed: int = 0) -> Path:
    """Write an 8x10 planted-blob dataset; groups mix positive and empty rows."""
    build_dataset(_TinySource(seed=seed), dest, BuildSpec(augment=AugmentRecipe(elements=("id",))))
    return dest


def _train_run(
    dataset: Path, run_root: Path, *, kind: str, epochs: int = 2, **overrides: object
) -> Path:
    fields: dict[str, object] = {
        "dataset": str(dataset),
        "run_dir": str(run_root),
        "run_id": "run",
        "kind": kind,
        "arch": "pactnet_w8_d2" if kind == "classifier" else "dilatenet_w8_d2",
        "device": "cpu",
        "epochs": epochs,
        "batch_size": 2,
    }
    fields.update(overrides)
    result = train(TrainConfig(**cast(Any, fields)))
    assert isinstance(result, Ok), result
    return result.value


def _cfg(run: Path, out: Path, **overrides: object) -> ModelAnalysisConfig:
    fields: dict[str, object] = {
        "run": str(run),
        "out": str(out),
        "batch_size": 2,
        "device": "cpu",
        "generalization": GeneralizationConfig(bootstrap_replicates=8),
        "plot": PlotConfig(formats=("png",), dpi=72, width_inches=5.0, height_inches=3.0),
    }
    fields.update(overrides)
    return ModelAnalysisConfig(**cast(Any, fields))


@pytest.fixture(scope="module")
def classifier_sources(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """One pristine tiny dataset and trained classifier run, shared read-only."""
    base = tmp_path_factory.mktemp("classifier")
    dataset = _dataset(base / "ds")
    run = _train_run(dataset, base / "runs", kind="classifier")
    return dataset, run


@pytest.fixture(scope="module")
def classifier_bundle(
    tmp_path_factory: pytest.TempPathFactory, classifier_sources: tuple[Path, Path]
) -> Path:
    """One frozen classifier analysis bundle, shared read-only."""
    _, run = classifier_sources
    out = tmp_path_factory.mktemp("classifier-bundle") / "analysis"
    result = analyze_model(_cfg(run, out))
    assert isinstance(result, Ok), result
    return out


@pytest.fixture(scope="module")
def segmentor_sources(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """One pristine tiny dataset and trained segmentor run, shared read-only."""
    base = tmp_path_factory.mktemp("segmentor")
    dataset = _dataset(base / "ds")
    run = _train_run(dataset, base / "runs", kind="segmentor")
    return dataset, run


@pytest.fixture(scope="module")
def segmentor_bundle(
    tmp_path_factory: pytest.TempPathFactory, segmentor_sources: tuple[Path, Path]
) -> Path:
    """One frozen segmentor analysis bundle, shared read-only."""
    _, run = segmentor_sources
    out = tmp_path_factory.mktemp("segmentor-bundle") / "analysis"
    result = analyze_model(_cfg(run, out))
    assert isinstance(result, Ok), result
    return out


def _manifest_hash(dataset: Path) -> str:
    return hashlib.sha256((dataset / "dataset.json").read_bytes()).hexdigest()


def _evaluating_fake(
    dataset: Path, series: tuple[float, ...], metric: str
) -> tuple[Callable[..., Result[SplitEvidence, str]], list[str]]:
    """Stub ``evaluate_split`` for the training loop; keeps the manifest hash."""
    values = iter(series)
    splits: list[str] = []

    def fake(
        model: torch.nn.Module,
        root: Path,
        manifest: DatasetManifest,
        cfg: EvaluationConfig,
        objective: PlumeLoss | None = None,
        capture: CaptureSink | None = None,
    ) -> Result[SplitEvidence, str]:
        splits.append(cfg.split)
        score = next(values) if cfg.split == "val" else 0.0
        return Ok(
            SplitEvidence(
                task=cfg.kind,
                split=cfg.split,
                dataset_hash=manifest.dataset_hash,
                dataset_manifest_hash=_manifest_hash(dataset),
                metrics=(
                    MetricValue(
                        name=metric,
                        value=score,
                        status="AVAILABLE",
                        support=MetricSupport(unit="IMAGE", n=1),
                    ),
                ),
            )
        )

    return fake, splits


def _spy_evaluate(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Wrap the measurement evaluator and record the evaluated splits."""
    calls: list[str] = []
    real = evaluate_mod.evaluate_split

    def spy(
        model: torch.nn.Module,
        root: Path,
        manifest: DatasetManifest,
        cfg: EvaluationConfig,
        objective: PlumeLoss | None = None,
        capture: CaptureSink | None = None,
    ) -> Result[SplitEvidence, str]:
        calls.append(cfg.split)
        return real(model, root, manifest, cfg, objective=objective, capture=capture)

    monkeypatch.setattr("tools.ml_models.analysis.model_measurement.evaluate_split", spy)
    return calls


def _verify_model_bundle(out: Path) -> ModelTrainingSummary:
    verified = verify_bundle(out)
    assert isinstance(verified, Ok), verified
    summary = verified.value
    assert isinstance(summary, ModelTrainingSummary)
    return summary


def _mutated_bundle(src: Path, dst: Path, path: str, transform: Callable[[bytes], bytes]) -> Path:
    """Copy a bundle, replace one file's bytes and keep all checksums self-consistent."""
    shutil.copytree(src, dst)
    new_bytes = transform((dst / path).read_bytes())
    (dst / path).write_bytes(new_bytes)
    decoded = decode_summary((dst / "summary.json").read_bytes())
    assert isinstance(decoded, Ok)
    old = decoded.value
    assert isinstance(old, ModelTrainingSummary)
    old_ref = next(ref for ref in old.artifacts if ref.path == path)
    new_ref = replace(
        old_ref, sha256=hashlib.sha256(new_bytes).hexdigest(), size_bytes=len(new_bytes)
    )
    swapped = tuple(new_ref if ref.path == path else ref for ref in old.artifacts)
    encoded = encode_summary(replace(old, artifacts=swapped))
    assert isinstance(encoded, Ok)
    (dst / "summary.json").write_bytes(encoded.value)
    return dst


def test_classifier_bundle_structure_and_split_metrics(classifier_bundle: Path) -> None:
    """The published bundle is complete, verifies, and persists every exact metric."""
    out = classifier_bundle
    summary = _verify_model_bundle(out)
    assert summary.status == "COMPLETE"
    assert summary.checkpoint.kind == "classifier"
    assert {split.split for split in summary.splits} == {"train", "val"}
    assert sum(1 for _ in out.rglob("summary.json")) == 1
    ref_paths = {ref.path for ref in summary.artifacts}
    for required in (
        "model-evidence.json",
        "config.toml",
        "rendering.json",
        "training-figure-data.json",
        "task/classifier/model-figure-data.json",
        "generalization/model-figure-data.json",
        "prediction-manifest.json",
        "tables/split_metrics.csv",
        "tables/split_metrics.parquet",
        "tables/split_metrics_schema.json",
        "source/config.toml",
        "source/execution.json",
        "source/history.jsonl",
        "source/training-dataset.json",
        "source/evaluation-dataset.json",
    ):
        assert required in ref_paths, required
        assert (out / required).is_file(), required
    assert not any(path.startswith("checkpoints/") for path in ref_paths)
    outputs = {record.name: record for record in summary.outputs}
    assert outputs["figures"].status == "AVAILABLE"
    with (out / "tables/split_metrics.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for column in (
        "split",
        "task",
        "population",
        "stratum_name",
        "stratum_value",
        "metric",
        "value",
        "status",
        "reason",
        "unit",
        "aggregation",
        "threshold",
        "support_unit",
        "support_n",
        "support_counts_json",
    ):
        assert column in rows[0]
    expected = sum(len(split.metrics) for split in summary.splits) + sum(
        len(stratum.metrics) for split in summary.splits for stratum in split.strata
    )
    assert len(rows) == expected
    measured = {
        (metric.name, metric.status) for split in summary.splits for metric in split.metrics
    }
    recorded = {(row["metric"], row["status"]) for row in rows if row["population"] == "cohort"}
    assert measured == recorded
    objective = {row["metric"] for row in rows if row["metric"].startswith("objective_")}
    assert objective == {"objective_loss", "objective_bce", "objective_focal", "objective_dice"}
    inactive = [
        row for row in rows if row["metric"] == "objective_focal" and row["population"] == "cohort"
    ]
    assert inactive and all(row["status"] == "UNAVAILABLE" and row["reason"] for row in inactive)


def test_segmentor_bundle_publishes(segmentor_bundle: Path) -> None:
    """A real tiny segmentor run analyzes into a verifiable bundle."""
    summary = _verify_model_bundle(segmentor_bundle)
    assert summary.checkpoint.kind == "segmentor"
    assert {split.split for split in summary.splits} == {"train", "val"}
    ref_paths = {ref.path for ref in summary.artifacts}
    assert "task/segmentor/model-figure-data.json" in ref_paths


def test_test_split_never_evaluated_by_default(
    classifier_sources: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An evaluator that refuses test proves it is never called by default."""
    _, run = classifier_sources
    calls = _spy_evaluate(monkeypatch)
    result = analyze_model(_cfg(run, tmp_path / "analysis"))
    assert isinstance(result, Ok), result
    assert "test" not in calls


def test_final_test_evaluates_the_selected_test_point_once(
    classifier_sources: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``final_test`` adds exactly one test evaluation on the selected checkpoint."""
    _, run = classifier_sources
    calls = _spy_evaluate(monkeypatch)
    out = tmp_path / "analysis"
    result = analyze_model(_cfg(run, out, final_test=True))
    assert isinstance(result, Ok), result
    summary = _verify_model_bundle(out)
    assert {split.split for split in summary.splits} == {"train", "val", "test"}
    assert calls.count("test") == 1
    assert all(split.checkpoint_hash == summary.checkpoint.sha256 for split in summary.splits)


def test_best_and_last_resolve_to_distinct_recorded_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``best`` selects the recorded best hash even when it differs from ``last``."""
    dataset = _dataset(tmp_path / "ds")
    fake, _ = _evaluating_fake(dataset, (0.8, 0.6, 0.7), "average_precision")
    monkeypatch.setattr(evaluate_mod, "evaluate_split", fake)
    run = _train_run(
        dataset,
        tmp_path / "runs",
        kind="classifier",
        epochs=3,
        eval_interval=1,
        val_metric="average_precision",
    )
    monkeypatch.undo()
    history = read_training_history(run)
    assert isinstance(history, Ok), history
    execution = history.value.execution
    assert execution.best is not None and execution.last is not None
    assert execution.best.identity.epoch == 1
    assert execution.last.identity.epoch == 3
    assert execution.best.identity.sha256 != execution.last.identity.sha256
    result = analyze_model(_cfg(run, tmp_path / "analysis"))
    assert isinstance(result, Ok), result
    summary = _verify_model_bundle(tmp_path / "analysis")
    assert summary.checkpoint == execution.best.identity


def test_mutated_checkpoint_bytes_rejected_before_evaluation(
    classifier_sources: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Changed checkpoint bytes fail the recorded-hash check before any evaluation."""
    _, run = classifier_sources
    copied = tmp_path / "run"
    shutil.copytree(run, copied)
    checkpoint = copied / "checkpoints" / "best.pt"
    data = bytearray(checkpoint.read_bytes())
    data[-1] ^= 0xFF
    checkpoint.write_bytes(bytes(data))
    calls = _spy_evaluate(monkeypatch)
    out = tmp_path / "analysis"
    result = analyze_model(_cfg(copied, out))
    assert isinstance(result, Err)
    assert "checkpoint" in result.error
    assert calls == []
    assert not out.exists()


def test_mutated_dataset_manifest_and_content_rejected(
    classifier_sources: tuple[Path, Path], tmp_path: Path
) -> None:
    """Changed manifest bytes or shard content fail verification, then restore."""
    dataset, run = classifier_sources
    manifest = dataset / "dataset.json"
    original = manifest.read_bytes()
    try:
        manifest.write_bytes(original + b" ")
        result = analyze_model(_cfg(run, tmp_path / "a1"))
        assert isinstance(result, Err)
        assert not (tmp_path / "a1").exists()
    finally:
        manifest.write_bytes(original)
    shard = next(path for path in dataset.rglob("*.npy"))
    original_shard = shard.read_bytes()
    try:
        shard.write_bytes(original_shard + b"\x00")
        result = analyze_model(_cfg(run, tmp_path / "a2"))
        assert isinstance(result, Err)
        assert not (tmp_path / "a2").exists()
    finally:
        shard.write_bytes(original_shard)


def test_source_mutation_during_load_rejected(
    classifier_sources: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A source changed mid-load is caught by the final source re-verification."""
    _, run = classifier_sources
    copied = tmp_path / "run"
    shutil.copytree(run, copied)
    config = copied / "config.toml"
    saved = config.read_bytes()
    import tools.ml_models.analysis.model_inputs as inputs_mod
    import tools.ml_models.train.provenance as provenance_mod

    real = provenance_mod.training_provenance

    def mutating(dataset: str | Path, manifest: DatasetManifest, task: str) -> dict[str, object]:
        config.write_bytes(saved + b"\n# tampered\n")
        return real(dataset, manifest, task)

    monkeypatch.setattr(inputs_mod, "training_provenance", mutating)
    result = analyze_model(_cfg(copied, tmp_path / "analysis"))
    assert isinstance(result, Err)
    assert "changed during input verification" in result.error
    assert not (tmp_path / "analysis").exists()


def test_config_toml_mismatch_rejected(
    classifier_sources: tuple[Path, Path], tmp_path: Path
) -> None:
    """A rewritten run config.toml disagrees with the recorded execution."""
    _, run = classifier_sources
    copied = tmp_path / "run"
    shutil.copytree(run, copied)
    config = copied / "config.toml"
    config.write_text(config.read_text(encoding="utf-8").replace("epochs = 2", "epochs = 3"))
    out = tmp_path / "analysis"
    result = analyze_model(_cfg(copied, out))
    assert isinstance(result, Err)
    assert "config" in result.error
    assert not out.exists()


def test_full_retention_captures_dense_predictions(
    classifier_sources: tuple[Path, Path], tmp_path: Path
) -> None:
    """``FULL`` retention adds dense prediction artifacts beside compact rows."""
    _, run = classifier_sources
    out = tmp_path / "analysis"
    result = analyze_model(_cfg(run, out, capture=CaptureConfig(retention="FULL")))
    assert isinstance(result, Ok), result
    summary = _verify_model_bundle(out)
    ref_paths = {ref.path for ref in summary.artifacts}
    assert any(path.startswith("capture/") and path.endswith("rows.jsonl") for path in ref_paths)
    assert any(
        path.startswith("capture/") and "/full/" in path and path.endswith(".npz")
        for path in ref_paths
    )


def test_segmentor_full_vs_compact_retention(
    segmentor_sources: tuple[Path, Path], tmp_path: Path
) -> None:
    """Segmentor ``FULL`` retention stores dense arrays; ``COMPACT`` stores none."""
    _, run = segmentor_sources
    full = analyze_model(_cfg(run, tmp_path / "full", capture=CaptureConfig(retention="FULL")))
    compact = analyze_model(
        _cfg(run, tmp_path / "compact", capture=CaptureConfig(retention="COMPACT"))
    )
    assert isinstance(full, Ok), full
    assert isinstance(compact, Ok), compact
    full_refs = {ref.path for ref in _verify_model_bundle(tmp_path / "full").artifacts}
    compact_refs = {ref.path for ref in _verify_model_bundle(tmp_path / "compact").artifacts}
    assert any("/full/" in path for path in full_refs)
    assert not any("/full/" in path for path in compact_refs)
    assert any(path.endswith("rows.jsonl") for path in compact_refs)


def test_metric_reference_records_explicit_unknowns(classifier_bundle: Path) -> None:
    """Unknown metric names keep explicit definition-free entries with reasons."""
    evidence = json.loads((classifier_bundle / "model-evidence.json").read_bytes())
    unknown = {
        entry["name"]: entry
        for entry in evidence["metric_reference"]
        if entry["definition"] is None
    }
    assert "objective_focal" in unknown
    assert unknown["objective_focal"]["reason"] == "unknown metric definition 'objective_focal'"


def test_missing_condition_figure_is_indexed_unavailable(classifier_bundle: Path) -> None:
    """A cohort without recorded conditions indexes the figure UNAVAILABLE with a reason."""
    summary = _verify_model_bundle(classifier_bundle)
    outputs = {record.name: record for record in summary.outputs}
    coverage = [
        record
        for name, record in outputs.items()
        if name.startswith("model_figure:generalization:") and "coverage_conditions" in name
    ]
    assert coverage
    assert all(
        record.status == "UNAVAILABLE" and record.reason == "no categorical conditions recorded"
        for record in coverage
    )
    assert outputs["generalization:val:conditions"].status == "UNAVAILABLE"


def test_external_evaluation_dataset_keeps_distinct_identity(
    classifier_sources: tuple[Path, Path], tmp_path: Path
) -> None:
    """An external eval dataset is measured while identities stay distinct."""
    _, run = classifier_sources
    external = _dataset(tmp_path / "ds2", seed=7)
    out = tmp_path / "analysis"
    result = analyze_model(_cfg(run, out, dataset=str(external)))
    assert isinstance(result, Ok), result
    summary = _verify_model_bundle(out)
    assert summary.evaluation_dataset != summary.training_dataset
    assert summary.checkpoint.training_dataset_hash == summary.training_dataset.content_hash


def test_render_twice_restyles_without_changing_science(
    classifier_bundle: Path, tmp_path: Path
) -> None:
    """Two renders with different plot configs keep every frozen value identical."""
    original = _verify_model_bundle(classifier_bundle)
    out_png = tmp_path / "render-png"
    out_svg = tmp_path / "render-svg"
    first = render_analysis(classifier_bundle, PlotConfig(formats=("png",), dpi=96), out_png)
    assert isinstance(first, Ok), first
    second = render_analysis(
        classifier_bundle,
        PlotConfig(formats=("svg",), dpi=72, width_inches=6.0, height_inches=4.0),
        out_svg,
    )
    assert isinstance(second, Ok), second
    re_rendered = _verify_model_bundle(out_png)
    re_rendered_svg = _verify_model_bundle(out_svg)
    for summary in (re_rendered, re_rendered_svg):
        assert summary.measurement_id == original.measurement_id
        assert summary.code == original.code
        assert summary.checkpoint == original.checkpoint
        assert summary.config_digest == original.config_digest
        assert summary.splits == original.splits
        assert summary.metrics == original.metrics
    frozen_names = {
        ref.path
        for ref in original.artifacts
        if ref.kind not in ("FIGURE", "VISUAL") and ref.path != "rendering.json"
    }
    for name in sorted(frozen_names):
        assert (out_png / name).read_bytes() == (classifier_bundle / name).read_bytes()
        assert (out_svg / name).read_bytes() == (classifier_bundle / name).read_bytes()
    assert any(
        ref.path.endswith(".svg") for ref in re_rendered_svg.artifacts if ref.kind == "FIGURE"
    )


def test_render_never_invokes_scientific_code(
    classifier_bundle: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Render must not call measurement, evaluation, model loading, or bootstrap code."""
    import tools.ml_models.analysis.metrics.generalization as generalization_mod
    import tools.ml_models.analysis.model_inputs as inputs_mod
    import tools.ml_models.analysis.model_summary as summary_mod

    def _boom(*args: object, **kwargs: object) -> object:
        raise AssertionError("scientific code ran during render")

    monkeypatch.setattr(measure_mod, "measure_model", _boom)
    monkeypatch.setattr("tools.ml_models.analysis.model_measurement.evaluate_split", _boom)
    monkeypatch.setattr(evaluate_mod, "evaluate_split", _boom)
    monkeypatch.setattr(inputs_mod, "load_model_inputs", _boom)
    monkeypatch.setattr(summary_mod, "model_summary", _boom)
    monkeypatch.setattr(generalization_mod, "measure_generalization", _boom)
    monkeypatch.setattr(generalization_mod, "fit_baseline", _boom)
    out = tmp_path / "rendered"
    rendered = render_analysis(classifier_bundle, PlotConfig(formats=("png",), dpi=72), out)
    assert isinstance(rendered, Ok), rendered


def test_render_corrupt_recipe_fails_with_no_output(
    classifier_bundle: Path, tmp_path: Path
) -> None:
    """A corrupt frozen recipe fails decode with an actionable error and no output."""
    mutated = _mutated_bundle(
        classifier_bundle,
        tmp_path / "mutated",
        "task/classifier/model-figure-data.json",
        lambda _: b'{"schema_version": 1, "figures": [{"bogus": true}]}',
    )
    out = tmp_path / "rendered"
    rendered = render_analysis(mutated, PlotConfig(formats=("png",), dpi=72), out)
    assert isinstance(rendered, Err)
    assert "fresh analyze required" in rendered.error
    assert not out.exists()


def test_render_rejects_self_consistent_but_altered_evidence(
    classifier_bundle: Path, tmp_path: Path
) -> None:
    """Checksummed edits to identities, config, or preview rows are still rejected."""
    cases: list[tuple[str, Path, Callable[[bytes], bytes]]] = []

    def swap_figure_identity(raw: bytes) -> bytes:
        payload = json.loads(raw)
        payload["figures"][0]["identity"]["dataset_hash"] = "0" * 64
        return json.dumps(payload).encode()

    cases.append(("task/classifier/model-figure-data.json", tmp_path / "fig", swap_figure_identity))

    def swap_figure_checkpoint(raw: bytes) -> bytes:
        payload = json.loads(raw)
        payload["figures"][0]["identity"]["checkpoint_hash"] = "2" * 64
        return json.dumps(payload).encode()

    cases.append(
        ("task/classifier/model-figure-data.json", tmp_path / "ckpt", swap_figure_checkpoint)
    )

    def swap_config(raw: bytes) -> bytes:
        return raw.replace(b"batch_size = 2", b"batch_size = 4")

    cases.append(("config.toml", tmp_path / "cfg", swap_config))

    def swap_doc(raw: bytes) -> bytes:
        payload = json.loads(raw)
        payload["baseline"]["training_dataset_hash"] = "1" * 64
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()

    cases.append(("model-evidence.json", tmp_path / "doc", swap_doc))

    def swap_row(raw: bytes) -> bytes:
        payload = json.loads(raw)
        payload["previews"][0]["row"]["key"]["tile_id"] = "forged"
        return json.dumps(payload).encode()

    cases.append(("prediction-manifest.json", tmp_path / "row", swap_row))

    for index, (path, dst, transform) in enumerate(cases):
        mutated = _mutated_bundle(classifier_bundle, dst, path, transform)
        out = tmp_path / f"out-{index}"
        rendered = render_analysis(mutated, PlotConfig(formats=("png",), dpi=72), out)
        assert isinstance(rendered, Err), path
        assert not out.exists(), path


def test_render_rejects_tampered_source_snapshots(classifier_bundle: Path, tmp_path: Path) -> None:
    """Source snapshot tampering fails typed validation even with valid checksums."""
    cases: list[tuple[str, Path, Callable[[bytes], bytes]]] = []

    def drop_snapshot(raw: bytes) -> bytes:
        payload = json.loads(raw)
        payload["source_snapshots"] = payload["source_snapshots"][:1]
        return json.dumps(payload).encode()

    cases.append(("model-evidence.json", tmp_path / "drop", drop_snapshot))

    def duplicate_snapshot(raw: bytes) -> bytes:
        payload = json.loads(raw)
        payload["source_snapshots"] = [*payload["source_snapshots"], payload["source_snapshots"][0]]
        return json.dumps(payload).encode()

    cases.append(("model-evidence.json", tmp_path / "dup", duplicate_snapshot))

    def bump_epoch(raw: bytes) -> bytes:
        payload = json.loads(raw)
        payload["best"]["identity"]["epoch"] += 1
        return json.dumps(payload).encode()

    cases.append(("source/execution.json", tmp_path / "epoch", bump_epoch))

    cases.append(("source/config.toml", tmp_path / "toml", lambda _: b"epochs = ["))

    def grow_manifest(raw: bytes) -> bytes:
        return raw + b" "

    cases.append(("source/training-dataset.json", tmp_path / "manifest", grow_manifest))

    for index, (path, dst, transform) in enumerate(cases):
        mutated = _mutated_bundle(classifier_bundle, dst, path, transform)
        out = tmp_path / f"snap-out-{index}"
        rendered = render_analysis(mutated, PlotConfig(formats=("png",), dpi=72), out)
        assert isinstance(rendered, Err), path
        assert not out.exists(), path


def test_linked_output_and_bundle_root_rejected(
    classifier_sources: tuple[Path, Path], classifier_bundle: Path, tmp_path: Path
) -> None:
    """Linked output ancestors and linked bundle roots fail closed, foreign data kept."""
    _, run = classifier_sources
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    marker = foreign / "keep.txt"
    marker.write_text("keep")
    link = tmp_path / "link"
    linked = False
    try:
        os.symlink(foreign, link, target_is_directory=True)
        linked = True
    except OSError:
        proc = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(foreign)], capture_output=True
        )
        linked = proc.returncode == 0
    if not linked:
        pytest.skip("no privilege to create a link for the output check")
    result = analyze_model(_cfg(run, link / "analysis"))
    assert isinstance(result, Err)
    assert not (foreign / "analysis").exists()
    assert marker.read_text() == "keep"
    rendered = render_analysis(classifier_bundle, PlotConfig(), link / "rendered")
    assert isinstance(rendered, Err)
    assert not (foreign / "rendered").exists()
    bundle_link = tmp_path / "bundle-link"
    linked_bundle = False
    try:
        os.symlink(classifier_bundle, bundle_link, target_is_directory=True)
        linked_bundle = True
    except OSError:
        proc = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(bundle_link), str(classifier_bundle)],
            capture_output=True,
        )
        linked_bundle = proc.returncode == 0
    if linked_bundle:
        assert isinstance(verify_bundle(bundle_link), Err)


def test_checkpoint_payload_rejects_bool_counters(
    classifier_sources: tuple[Path, Path],
) -> None:
    """Serialized counters must be real nonnegative ints, never bools."""
    import tools.ml_models.analysis.model_inputs as inputs_mod

    _, run = classifier_sources
    history = read_training_history(run)
    assert isinstance(history, Ok), history
    execution = history.value.execution
    record = execution.best
    assert record is not None
    payload: dict[str, object] = {
        "kind": "classifier",
        "arch": "pactnet_w8_d2",
        "state_dict": {},
        "epoch": True,
        "step": 0,
        "samples_seen": 0,
        "dataset_hash": record.identity.training_dataset_hash,
        "conditioning": execution.conditioning,
        "config": {},
        "provenance": {},
        "metric": execution.metric,
        "direction": "MAXIMIZE",
        "value": 0.5,
        "validation": {},
    }
    result = inputs_mod._verify_checkpoint_payload(
        payload, record, history.value.records, execution
    )
    assert isinstance(result, Err)
    assert "epoch" in result.error


def test_render_inside_bundle_and_recorded_sources_rejected(
    classifier_sources: tuple[Path, Path], classifier_bundle: Path, tmp_path: Path
) -> None:
    """Render refuses output inside the bundle, the run, and the source dataset."""
    dataset, run = classifier_sources
    for forbidden in (classifier_bundle / "nested", run / "rendered", dataset / "rendered"):
        result = render_analysis(classifier_bundle, PlotConfig(), forbidden)
        assert isinstance(result, Err), forbidden
        assert not forbidden.exists()


def test_dataset_render_refuses_output_inside_source_dataset(tmp_path: Path) -> None:
    """A dataset bundle cannot be re-rendered into its recorded source dataset."""
    dataset = _dataset(tmp_path / "ds")
    bundle = tmp_path / "analysis"
    cfg = DatasetAnalysisConfig(
        dataset=str(dataset),
        out=str(bundle),
        plot=PlotConfig(formats=("png",), dpi=72),
        capture=CaptureConfig(max_preview_images=0, examples_per_family=0),
    )
    published = analyze_dataset(cfg)
    assert isinstance(published, Ok), published
    result = render_analysis(bundle, PlotConfig(), dataset / "rendered")
    assert isinstance(result, Err)
    assert not (dataset / "rendered").exists()
    inside = render_analysis(bundle, PlotConfig(), bundle / "nested")
    assert isinstance(inside, Err)
    assert not (bundle / "nested").exists()


def test_render_incomplete_marker_rejected(classifier_bundle: Path, tmp_path: Path) -> None:
    """A bundle retaining the in-progress marker is refused."""
    marked = tmp_path / "marked"
    shutil.copytree(classifier_bundle, marked)
    (marked / INCOMPLETE_FILENAME).write_text("in progress")
    rendered = render_analysis(marked, PlotConfig(), tmp_path / "out")
    assert isinstance(rendered, Err)
    assert not (tmp_path / "out").exists()


def test_existing_output_and_linked_checkpoint_rejected(
    classifier_sources: tuple[Path, Path], tmp_path: Path
) -> None:
    """Preflight refuses preexisting outputs and linked source components."""
    _, run = classifier_sources
    out = tmp_path / "analysis"
    out.mkdir()
    result = analyze_model(_cfg(run, out))
    assert isinstance(result, Err)
    copied = tmp_path / "run"
    shutil.copytree(run, copied)
    shutil.move(str(copied / "checkpoints"), str(tmp_path / "checkpoints-real"))
    linked = False
    try:
        os.symlink(tmp_path / "checkpoints-real", copied / "checkpoints")
        linked = True
    except OSError:
        proc = subprocess.run(
            [
                "cmd",
                "/c",
                "mklink",
                "/J",
                str(copied / "checkpoints"),
                str(tmp_path / "checkpoints-real"),
            ],
            capture_output=True,
        )
        linked = proc.returncode == 0
    if not linked:
        pytest.skip("no privilege to create a link for the source check")
    rejected = analyze_model(_cfg(copied, tmp_path / "linked-out"))
    assert isinstance(rejected, Err)
    assert not (tmp_path / "linked-out").exists()


def test_publication_failure_leaves_incomplete_marker(
    classifier_sources: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mid-publication write failure keeps ``.incomplete`` so verify refuses it."""
    _, run = classifier_sources
    out = tmp_path / "analysis"
    real_open = Path.open

    def flaky(
        self: Path,
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> object:
        if "b" in mode and self.name == "model-evidence.json" and str(self).startswith(str(out)):
            raise OSError("injected publication failure")
        return real_open(self, mode, buffering, encoding, errors, newline)

    monkeypatch.setattr(Path, "open", flaky)
    result = analyze_model(_cfg(run, out))
    assert isinstance(result, Err)
    assert (out / INCOMPLETE_FILENAME).exists()
    assert isinstance(verify_bundle(out), Err)


def test_unknown_checkpoint_selector_rejected(
    classifier_sources: tuple[Path, Path], tmp_path: Path
) -> None:
    """Selectors outside the recorded best/last records fail closed."""
    _, run = classifier_sources
    result = analyze_model(_cfg(run, tmp_path / "out", checkpoint="epoch2"))
    assert isinstance(result, Err)
    assert not (tmp_path / "out").exists()


@pytest.mark.slow
def test_default_size_bundles_render(
    classifier_sources: tuple[Path, Path],
    segmentor_sources: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    """Default ``PlotConfig`` bundles render representative full-size figures."""
    for index, (_, run) in enumerate((classifier_sources, segmentor_sources)):
        out = tmp_path / f"analysis-{index}"
        result = analyze_model(_cfg(run, out, plot=PlotConfig()))
        assert isinstance(result, Ok), result
        summary = _verify_model_bundle(out)
        assert any(ref.kind == "FIGURE" and ref.path.endswith(".png") for ref in summary.artifacts)
