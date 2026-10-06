"""Tests for frozen training-figure artifact serialization."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import tools.ml_models.analysis.training_artifacts as artifact_module
from flight.libs.types import Err, Ok, Result
from tools.ml_models.analysis.artifacts import Scalar, read_table
from tools.ml_models.analysis.contracts import ArtifactRef
from tools.ml_models.analysis.training_artifacts import (
    TRAINING_POINTS_SCHEMA,
    TrainingArtifacts,
    training_artifacts,
)
from tools.ml_models.analysis.training_figures import (
    TrainingFigure,
    TrainingMarker,
    TrainingSeries,
)


def _canonical(data: object) -> bytes:
    return json.dumps(data, sort_keys=True, allow_nan=False, separators=(",", ":")).encode("utf-8")


def _figures() -> tuple[TrainingFigure, ...]:
    return (
        TrainingFigure(
            "optimization_loss_loglog",
            "Optimization objective per attempted batch",
            "Successful optimizer updates",
            "Configured weighted objective",
            (
                TrainingSeries(
                    "optimization batch objective",
                    (1.0, 2.0, 2.0, 3.0),
                    (3.0, 0.0, 1.5, 1.0),
                    5,
                    "attempted optimization batches; resampled exposures",
                ),
            ),
            (
                TrainingMarker("best checkpoint", 2.0, "a" * 64),
                TrainingMarker("stopping state", 3.0),
            ),
            "COMPLETED",
            "log",
            "log",
        ),
        TrainingFigure(
            "selected_final",
            "Comparable canonical evaluation objective",
            "Successful optimizer updates",
            "Configured weighted objective",
            (
                TrainingSeries("validation", (2.0, 3.0), (2.0, None), 4, "captured val"),
                TrainingSeries(
                    "selected-checkpoint test",
                    (2.0,),
                    (3.0,),
                    2,
                    "one selected-checkpoint test evaluation",
                    points_only=True,
                ),
            ),
            (),
            "COMPLETED",
            "linear",
            "linear",
        ),
        TrainingFigure(
            "amp_scale",
            "Recorded AMP scaler telemetry",
            "Successful optimizer updates",
            "AMP scale",
            (
                TrainingSeries(
                    "before update",
                    (),
                    (),
                    0,
                    "recorded AMP events",
                    exposure_unit="BATCH",
                ),
            ),
            (),
            "INTERRUPTED",
            "linear",
            "linear",
            "Requested values were not captured or are inactive",
        ),
    )


def _files(result: Result[TrainingArtifacts, str]) -> dict[str, bytes]:
    assert isinstance(result, Ok)
    return {file.path: file.data for file in result.value.files}


def test_bundle_files_references_and_frozen_payloads() -> None:
    figures = _figures()
    result = training_artifacts(figures)
    assert isinstance(result, Ok)
    files = _files(result)
    assert set(files) == {
        "training-figure-data.json",
        "tables/training_points.csv",
        "tables/training_points.parquet",
        "tables/training_schema.json",
    }
    assert files["training-figure-data.json"] == _canonical(
        {"schema_version": 1, "figures": [asdict(record) for record in figures]}
    )
    refs = {ref.path: ref for ref in result.value.references}
    assert set(refs) == set(files)
    for path, ref in refs.items():
        assert ref.sha256 == hashlib.sha256(files[path]).hexdigest()
        assert ref.size_bytes == len(files[path])
    schemas = json.loads(files["tables/training_schema.json"])
    expected_schema = json.loads(_canonical(asdict(TRAINING_POINTS_SCHEMA)))
    assert schemas == {
        "tables/training_points.csv": expected_schema,
        "tables/training_points.parquet": expected_schema,
    }
    n_points = sum(len(series.x) for record in figures for series in record.series)
    assert refs["tables/training_points.csv"].rows == n_points
    assert refs["tables/training_points.parquet"].rows == n_points


def test_point_rows_copy_raw_values_and_formats_agree(tmp_path: Path) -> None:
    figures = _figures()
    files = _files(training_artifacts(figures))
    parsed: dict[str, tuple[dict[str, Scalar], ...]] = {}
    for suffix in ("csv", "parquet"):
        target = tmp_path / f"points.{suffix}"
        target.write_bytes(files[f"tables/training_points.{suffix}"])
        rows = read_table(target, TRAINING_POINTS_SCHEMA)
        assert isinstance(rows, Ok)
        parsed[suffix] = rows.value
    assert parsed["csv"] == parsed["parquet"]
    expected: list[dict[str, Scalar]] = []
    for record in figures:
        for series in record.series:
            expected += [
                {
                    "figure": record.identifier,
                    "series": series.name,
                    "point_index": index,
                    "x": x,
                    "y": y,
                    "n_exposures": series.n_exposures,
                    "exposure_unit": series.exposure_unit,
                    "population": series.population,
                    "points_only": series.points_only,
                }
                for index, (x, y) in enumerate(zip(series.x, series.y, strict=True))
            ]
    assert list(parsed["csv"]) == expected
    assert any(row["y"] == 0.0 for row in parsed["csv"])
    assert any(row["y"] is None for row in parsed["csv"])
    assert any(row["points_only"] for row in parsed["csv"])
    assert (
        sum(
            1
            for row in parsed["csv"]
            if row["x"] == 2.0 and row["figure"] == "optimization_loss_loglog"
        )
        == 2
    )


def test_prefix_namespaces_paths_without_changing_bytes() -> None:
    figures = _figures()
    plain = _files(training_artifacts(figures))
    prefixed_result = training_artifacts(figures, prefix="evidence/loss")
    assert isinstance(prefixed_result, Ok)
    prefixed = _files(prefixed_result)
    assert all(path.startswith("evidence/loss/") for path in prefixed)
    assert {p.removeprefix("evidence/loss/"): b for p, b in prefixed.items()} == plain
    plain_result = training_artifacts(figures)
    assert isinstance(plain_result, Ok)
    plain_refs = {ref.path: ref for ref in plain_result.value.references}
    prefixed_refs = {
        ref.path.removeprefix("evidence/loss/"): ref for ref in prefixed_result.value.references
    }
    assert {path: ref.sha256 for path, ref in prefixed_refs.items()} == {
        path: ref.sha256 for path, ref in plain_refs.items()
    }


@pytest.mark.parametrize("prefix", ["../escape", "/absolute", r"back\slash", "a//b"])
def test_prefix_rejects_unsafe_namespaces(prefix: str) -> None:
    assert isinstance(training_artifacts(_figures(), prefix=prefix), Err)


def _forced_table_failure(*args: object, **kwargs: object) -> Result[ArtifactRef, str]:
    return Err("forced codec failure")


def test_table_codec_failure_returns_err(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(artifact_module, "write_table", _forced_table_failure)
    result = training_artifacts(_figures())
    assert isinstance(result, Err)
    assert "forced codec failure" in result.error


def test_writer_never_reads_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    import tools.ml_models.analysis.training as training_module

    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("artifact writer must not read run history")

    monkeypatch.setattr(training_module, "read_training_history", _boom)
    assert isinstance(training_artifacts(_figures()), Ok)
