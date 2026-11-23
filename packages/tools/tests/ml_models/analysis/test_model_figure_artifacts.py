"""Frozen model-figure artifact serialization checks."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest
from flight.libs.types import Err, Ok, Result
from tools.ml_models.analysis.artifacts import read_table
from tools.ml_models.analysis.contracts import ArtifactRef, MetricSupport
from tools.ml_models.analysis.model_figure_artifacts import (
    MODEL_POINTS_SCHEMA,
    ModelFigureArtifacts,
    model_figure_artifacts,
)
from tools.ml_models.analysis.model_figures import (
    FigureIdentity,
    ModelFigure,
    ModelPoint,
    ModelSeries,
)

_IDENTITY = FigureIdentity(
    dataset_hash="a" * 64,
    dataset_manifest_hash="b" * 64,
    checkpoint_hash="c" * 64,
    task="classifier",
    split="val",
)
_SUPPORT = MetricSupport(unit="IMAGE", n=4)


def _figures() -> tuple[ModelFigure, ...]:
    series = ModelSeries(
        "precision_recall",
        (0.0, 0.5, 1.0),
        (1.0, None, 0.5),
        _SUPPORT,
        "PRE",
        lower=(0.9, None, 0.4),
        upper=(1.0, None, 0.6),
        point_support=(2, 0, 2),
    )
    matrix = ModelFigure(
        "confusion_truth_normalized",
        _IDENTITY,
        "Captured matrix",
        "Predicted class",
        "Truth class",
        "val images",
        kind="MATRIX",
        x_categories=("negative", "positive"),
        y_categories=("negative", "positive"),
        matrix=((0.5, 0.5), (None, None)),
        matrix_support=((1, 1), (0, 0)),
        matrix_range=(0.0, 1.0),
        reason=None,
        notes=("note",),
    )
    return (
        ModelFigure(
            "precision_recall",
            _IDENTITY,
            "Captured precision recall",
            "recall",
            "precision",
            "val image cohort",
            series=(series,),
            points=(ModelPoint("op", 0.3, 0.5),),
        ),
        matrix,
        ModelFigure(
            "roc",
            _IDENTITY,
            "Captured roc",
            "fpr",
            "tpr",
            "val image cohort",
            reason="ROC requires both truth classes",
        ),
    )


def _call(figures: tuple[ModelFigure, ...] | None = None, prefix: str = "") -> ModelFigureArtifacts:
    result = model_figure_artifacts(figures or _figures(), prefix=prefix)
    assert isinstance(result, Ok)
    return result.value


def test_manifest_is_byte_identical_to_frozen_recipes() -> None:
    artifacts = _call()
    manifest = next(file for file in artifacts.files if file.path == "model-figure-data.json")
    figures = _figures()
    expected = json.dumps(
        {"schema_version": 1, "figures": [asdict(figure) for figure in figures]},
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode()
    assert manifest.data == expected
    ref = next(ref for ref in artifacts.references if ref.path == "model-figure-data.json")
    assert ref.kind == "REFERENCE" and ref.format == "json"
    import hashlib

    assert ref.sha256 == hashlib.sha256(expected).hexdigest()


def test_point_rows_copy_exact_frozen_fields() -> None:
    artifacts = _call()
    csv = next(file for file in artifacts.files if file.path == "tables/model_points.csv")
    parquet = next(file for file in artifacts.files if file.path == "tables/model_points.parquet")
    schema_file = next(file for file in artifacts.files if file.path == "tables/model_schema.json")
    declared = json.loads(schema_file.data.decode())
    assert "tables/model_points.csv" in declared and "tables/model_points.parquet" in declared
    names = [c["name"] for c in declared["tables/model_points.csv"]["columns"]]
    assert names == [
        "figure",
        "task",
        "split",
        "dataset_hash",
        "dataset_manifest_hash",
        "checkpoint_hash",
        "series",
        "point_index",
        "x",
        "y",
        "lower",
        "upper",
        "support_unit",
        "support_n",
        "point_support_n",
        "style",
    ]

    import tempfile

    with tempfile.TemporaryDirectory() as staging:
        csv_path = Path(staging) / "points.csv"
        csv_path.write_bytes(csv.data)
        parquet_path = Path(staging) / "points.parquet"
        parquet_path.write_bytes(parquet.data)
        csv_rows = read_table(csv_path, MODEL_POINTS_SCHEMA)
        parquet_rows = read_table(parquet_path, MODEL_POINTS_SCHEMA)
    assert isinstance(csv_rows, Ok) and isinstance(parquet_rows, Ok)
    assert csv_rows.value == parquet_rows.value
    rows = csv_rows.value
    assert len(rows) == 3
    first = rows[0]
    assert first["figure"] == "precision_recall"
    assert first["task"] == "classifier" and first["split"] == "val"
    assert first["dataset_hash"] == "a" * 64
    assert first["dataset_manifest_hash"] == "b" * 64
    assert first["checkpoint_hash"] == "c" * 64
    assert first["series"] == "precision_recall"
    assert first["point_index"] == 0 and first["x"] == 0.0 and first["y"] == 1.0
    assert first["lower"] == 0.9 and first["upper"] == 1.0
    assert first["support_unit"] == "IMAGE" and first["support_n"] == 4
    assert first["point_support_n"] == 2 and first["style"] == "PRE"
    assert rows[1]["y"] is None and rows[1]["lower"] is None
    assert rows[1]["point_support_n"] == 0


def test_references_cover_every_file_with_exact_checksums() -> None:
    artifacts = _call()
    import hashlib

    by_path = {file.path: file for file in artifacts.files}
    assert {ref.path for ref in artifacts.references} == set(by_path)
    for ref in artifacts.references:
        assert ref.sha256 == hashlib.sha256(by_path[ref.path].data).hexdigest()
        assert ref.size_bytes == len(by_path[ref.path].data)
    formats = {ref.path: ref.format for ref in artifacts.references}
    assert formats["tables/model_points.csv"] == "csv"
    assert formats["tables/model_points.parquet"] == "parquet"
    for ref in artifacts.references:
        expected = "TABLE" if ref.path.startswith("tables/model_points.") else "REFERENCE"
        assert ref.kind == expected
    assert not any(file.path.endswith("summary.json") for file in artifacts.files)


def test_prefix_namespaces_keep_bytes_and_checksums() -> None:
    default = _call()
    prefixed = _call(prefix="evidence/train")
    by_path = {file.path: file.data for file in default.files}
    for file in prefixed.files:
        assert file.path.startswith("evidence/train/")
        assert file.data == by_path[file.path.removeprefix("evidence/train/")]
    assert {ref.path for ref in prefixed.references} == {
        "evidence/train/" + ref.path for ref in default.references
    }
    assert {ref.sha256 for ref in prefixed.references} == {ref.sha256 for ref in default.references}


@pytest.mark.parametrize(
    "prefix",
    ("../escape", "/absolute", r"back\slash", "a//b", "trail. "),
)
def test_unsafe_prefix_rejected(prefix: str) -> None:
    assert isinstance(model_figure_artifacts(_figures(), prefix=prefix), Err)


def test_misaligned_series_lengths_return_err() -> None:
    base = ModelFigure(
        "precision_recall",
        _IDENTITY,
        "Captured precision recall",
        "recall",
        "precision",
        "val image cohort",
        series=(
            ModelSeries("bad", (0.0, 1.0), (0.5, 0.5), _SUPPORT, lower=(0.1,), upper=(0.2, 0.4)),
        ),
    )
    assert isinstance(model_figure_artifacts((base,)), Err)
    short_support = ModelFigure(
        "precision_recall",
        _IDENTITY,
        "Captured precision recall",
        "recall",
        "precision",
        "val image cohort",
        series=(ModelSeries("bad", (0.0, 1.0), (0.5, 0.5), _SUPPORT, point_support=(1,)),),
    )
    assert isinstance(model_figure_artifacts((short_support,)), Err)


def test_table_codec_failure_returns_err(monkeypatch: pytest.MonkeyPatch) -> None:
    def _forced_table_failure(*args: object, **kwargs: object) -> Result[ArtifactRef, str]:
        return Err("forced codec failure")

    import tools.ml_models.analysis.model_figure_artifacts as module

    monkeypatch.setattr(module, "write_table", _forced_table_failure)
    result = model_figure_artifacts(_figures())
    assert isinstance(result, Err)
    assert "forced codec failure" in result.error


def test_helper_never_reads_sources_or_derives_figures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _bomb(*args: object, **kwargs: object) -> None:
        raise AssertionError("helper touched source/selection helpers")

    import tools.ml_models.analysis.classifier_figures as classifier_module
    import tools.ml_models.analysis.prediction_selections as selections

    monkeypatch.setattr(classifier_module, "classifier_figure_data", _bomb)
    monkeypatch.setattr(selections, "prediction_gallery_data", _bomb)
    assert isinstance(model_figure_artifacts(_figures()), Ok)
