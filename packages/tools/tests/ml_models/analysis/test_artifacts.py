"""Tests for artifact codecs, checksums, publication, and typed tables."""

import hashlib
import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest
from flight.libs.types import Err, Ok
from tools.ml_models.analysis.artifacts import (
    BundleFile,
    ColumnSpec,
    Scalar,
    TableSchema,
    artifact_ref,
    checksum_file,
    dataset_identity,
    publish_bundle,
    read_table,
    verify_bundle,
    write_table,
)
from tools.ml_models.analysis.contracts import (
    ArtifactKind,
    ArtifactRef,
    CheckpointIdentity,
    CodeIdentity,
    DatasetIdentity,
    SplitEvidence,
)
from tools.ml_models.analysis.summaries import DatasetSummary, ModelTrainingSummary

_HEX_A = "a" * 64
_HEX_B = "b" * 64
_HEX_C = "c" * 64
_HEX_D = "d" * 64
_HEX_E = "e" * 64


def _identity() -> DatasetIdentity:
    return DatasetIdentity(
        content_hash=_HEX_B,
        manifest_hash=_HEX_C,
        schema_version=2,
        source="fixture",
        band_names=("red",),
        gsd_reference_m=0.05,
    )


def _summary(**overrides: object) -> DatasetSummary:
    values: dict[str, object] = {
        "measurement_id": _HEX_A,
        "dataset": _identity(),
        "code": CodeIdentity(revision=None, dirty=None, reason="git unavailable"),
        "config_digest": _HEX_E,
    }
    values.update(overrides)
    return DatasetSummary(**values)  # type: ignore[arg-type]


def _ref_for(
    path: str, data: bytes, *, kind: ArtifactKind = "TABLE", fmt: str = "csv"
) -> ArtifactRef:
    ref = artifact_ref(path, data, kind=kind, format=fmt)
    assert isinstance(ref, Ok)
    return ref.value


def test_checksum_file_streamed(tmp_path: Path) -> None:
    """checksum_file returns the SHA-256 of the file bytes."""
    path = tmp_path / "blob.bin"
    data = b"\x00\x01" * 4096 + b"tail"
    path.write_bytes(data)
    result = checksum_file(path)
    assert isinstance(result, Ok)
    assert result.value == hashlib.sha256(data).hexdigest()
    missing = checksum_file(tmp_path / "none.bin")
    assert isinstance(missing, Err)
    directory = checksum_file(tmp_path)
    assert isinstance(directory, Err)


def test_dataset_identity_from_manifest(
    build_synthetic_dataset: Callable[..., Path], tmp_path: Path
) -> None:
    """dataset_identity binds content and manifest hashes without mutation."""
    root = build_synthetic_dataset(tmp_path / "ds")
    manifest_path = root / "dataset.json"
    before_bytes = manifest_path.read_bytes()
    result = dataset_identity(root)
    assert isinstance(result, Ok)
    identity = result.value
    assert identity.manifest_hash == hashlib.sha256(before_bytes).hexdigest()
    assert len(identity.content_hash) == 64
    assert identity.manifest_hash != identity.content_hash
    assert identity.schema_version == 2
    assert identity.source == "fixture"
    assert identity.gsd_reference_m > 0
    assert manifest_path.read_bytes() == before_bytes
    after = dataset_identity(root)
    assert isinstance(after, Ok)
    assert after.value.content_hash == identity.content_hash


def test_dataset_identity_rejects_bad_roots(tmp_path: Path) -> None:
    """Missing or corrupt manifests fail closed."""
    assert isinstance(dataset_identity(tmp_path / "missing"), Err)
    root = tmp_path / "ds"
    root.mkdir()
    (root / "dataset.json").write_text("{}", encoding="utf-8")
    assert isinstance(dataset_identity(root), Err)
    assert isinstance(dataset_identity(tmp_path / "ds" / "dataset.json"), Err)


def test_artifact_ref_derives_checksum_and_size(tmp_path: Path) -> None:
    """The factory computes hash and size from the actual bytes."""
    data = b"abc123"
    ref = artifact_ref("tables/t.csv", data, kind="TABLE", format="csv", rows=2)
    assert isinstance(ref, Ok)
    assert ref.value.sha256 == hashlib.sha256(data).hexdigest()
    assert ref.value.size_bytes == len(data)
    assert ref.value.rows == 2
    bad = artifact_ref("a/../t.csv", data, kind="TABLE", format="csv")
    assert isinstance(bad, Err)


def test_bundle_file_path_safety() -> None:
    """BundleFile enforces safe relative POSIX paths."""
    assert BundleFile(path="a/b.bin", data=b"x").path == "a/b.bin"
    for bad in ("../x", "a\\b", "/x", "C:/x", "a//b", ""):
        with pytest.raises(ValueError):
            BundleFile(path=bad, data=b"x")


def _publish(tmp_path: Path, summary: DatasetSummary, files: tuple[BundleFile, ...]) -> Path:
    out = tmp_path / "bundle"
    result = publish_bundle(out, summary, files)
    assert isinstance(result, Ok)
    return out


def test_publish_and_verify_roundtrip(tmp_path: Path) -> None:
    """A complete bundle publishes and verifies."""
    data = b"col\n1\n"
    ref = _ref_for("tables/m.csv", data)
    summary = _summary(artifacts=(ref,))
    out = _publish(tmp_path, summary, (BundleFile(path="tables/m.csv", data=data),))
    assert (out / "summary.json").is_file()
    assert not (out / ".incomplete").exists()
    verified = verify_bundle(out)
    assert isinstance(verified, Ok)
    assert verified.value == summary


def test_publish_refuses_existing_output(tmp_path: Path) -> None:
    """A second publisher fails without touching prior bytes."""
    data = b"v1"
    ref = _ref_for("m.csv", data)
    out = _publish(tmp_path, _summary(artifacts=(ref,)), (BundleFile(path="m.csv", data=data),))
    again = publish_bundle(out, _summary(artifacts=(ref,)), (BundleFile(path="m.csv", data=b"v2"),))
    assert isinstance(again, Err)
    assert (out / "m.csv").read_bytes() == b"v1"
    raced = tmp_path / "raced"
    raced.mkdir()
    result = publish_bundle(raced, _summary(), ())
    assert isinstance(result, Err)
    assert list(raced.iterdir()) == []


def test_publish_rejects_reference_mismatches(tmp_path: Path) -> None:
    """Refs must match supplied bytes exactly; nothing is reserved on failure."""
    ref = _ref_for("m.csv", b"real")
    out = tmp_path / "out"
    missing = publish_bundle(out, _summary(artifacts=(ref,)), ())
    assert isinstance(missing, Err)
    assert not out.exists()
    wrong = publish_bundle(
        out, _summary(artifacts=(ref,)), (BundleFile(path="m.csv", data=b"other"),)
    )
    assert isinstance(wrong, Err)
    assert not out.exists()
    orphan = publish_bundle(out, _summary(), (BundleFile(path="unreferenced.bin", data=b"x"),))
    assert isinstance(orphan, Err)
    assert not out.exists()


def test_publish_rejects_non_complete_summaries(tmp_path: Path) -> None:
    """Only COMPLETE summaries publish; markers still block verification."""
    for status in ("PARTIAL", "FAILED"):
        summary = _summary(status=status)
        out = tmp_path / f"out-{status.lower()}"
        result = publish_bundle(out, summary, ())
        assert isinstance(result, Err)
        assert not out.exists()
    out = _publish(tmp_path, _summary(), ())
    (out / ".incomplete").write_bytes(b"")
    assert isinstance(verify_bundle(out), Err)


def test_publish_refuses_source_dataset_root(
    build_synthetic_dataset: Callable[..., Path], tmp_path: Path
) -> None:
    """Outputs cannot land inside a hashed dataset root."""
    dataset_root = build_synthetic_dataset(tmp_path / "ds")
    inside = publish_bundle(dataset_root / "analysis", _summary(), (), dataset_root=dataset_root)
    assert isinstance(inside, Err)
    exact = publish_bundle(dataset_root, _summary(), (), dataset_root=dataset_root)
    assert isinstance(exact, Err)
    outside = publish_bundle(tmp_path / "out", _summary(), (), dataset_root=dataset_root)
    assert isinstance(outside, Ok)


def test_verify_bundle_rejects_corrupt_and_missing(tmp_path: Path) -> None:
    """Verify checks checksum, size, and existence of every reference."""
    data = b"col\n1\n"
    ref = _ref_for("m.csv", data)
    out = _publish(tmp_path, _summary(artifacts=(ref,)), (BundleFile(path="m.csv", data=data),))
    (out / "m.csv").write_bytes(b"corrupted")
    assert isinstance(verify_bundle(out), Err)
    (out / "m.csv").unlink()
    assert isinstance(verify_bundle(out), Err)
    no_summary = tmp_path / "empty"
    no_summary.mkdir()
    assert isinstance(verify_bundle(no_summary), Err)
    assert isinstance(verify_bundle(tmp_path / "absent"), Err)


def test_verify_bundle_rejects_split_and_history_refs(tmp_path: Path) -> None:
    """Split artifact refs and model-summary history are verified too."""
    data = b"x"
    ref = _ref_for("curves/c.csv", data)
    split = SplitEvidence(task="segmentor", split="test", dataset_hash=_HEX_B, artifacts=(ref,))
    summary = _summary(splits=(split,))
    out = _publish(tmp_path, summary, (BundleFile(path="curves/c.csv", data=data),))
    (out / "curves" / "c.csv").write_bytes(b"bad")
    assert isinstance(verify_bundle(out), Err)

    history_ref = _ref_for("history.jsonl", data, kind="REFERENCE", fmt="jsonl")
    model = _model_summary(history=history_ref)
    out2 = tmp_path / "model-bundle"
    published = publish_bundle(out2, model, (BundleFile(path="history.jsonl", data=data),))
    assert isinstance(published, Ok)
    (out2 / "history.jsonl").unlink()
    assert isinstance(verify_bundle(out2), Err)


def _model_summary(**overrides: object) -> ModelTrainingSummary:
    values: dict[str, object] = {
        "measurement_id": _HEX_A,
        "training_dataset": _identity(),
        "evaluation_dataset": _identity(),
        "checkpoint": CheckpointIdentity(
            sha256=_HEX_D,
            kind="segmentor",
            arch="u",
            training_dataset_hash=_HEX_B,
        ),
        "code": CodeIdentity(revision=None, dirty=None, reason="git unavailable"),
        "config_digest": _HEX_E,
    }
    values.update(overrides)
    return ModelTrainingSummary(**values)  # type: ignore[arg-type]


def test_shared_artifact_refs_deduplicate(tmp_path: Path) -> None:
    """The same ArtifactRef may be indexed at root and in split/history."""
    data = b"shared"
    ref = _ref_for("shared/t.csv", data)
    split = SplitEvidence(task="segmentor", split="test", dataset_hash=_HEX_B, artifacts=(ref,))
    summary = _summary(artifacts=(ref,), splits=(split,))
    out = _publish(tmp_path, summary, (BundleFile(path="shared/t.csv", data=data),))
    verified = verify_bundle(out)
    assert isinstance(verified, Ok)
    assert verified.value == summary

    history_ref = _ref_for("history/h.jsonl", data, kind="REFERENCE", fmt="jsonl")
    model = _model_summary(artifacts=(history_ref,), history=history_ref)
    out2 = tmp_path / "model"
    published = publish_bundle(out2, model, (BundleFile(path="history/h.jsonl", data=data),))
    assert isinstance(published, Ok)
    verified2 = verify_bundle(out2)
    assert isinstance(verified2, Ok)
    assert verified2.value == model


def test_conflicting_refs_same_path_rejected(tmp_path: Path) -> None:
    """Same path with different hash/size fails before any output is made."""
    ref_a = _ref_for("t.csv", b"a")
    ref_b = _ref_for("t.csv", b"bb")
    split = SplitEvidence(task="segmentor", split="test", dataset_hash=_HEX_B, artifacts=(ref_b,))
    summary = _summary(artifacts=(ref_a,), splits=(split,))
    out = tmp_path / "conflict"
    result = publish_bundle(out, summary, (BundleFile(path="t.csv", data=b"a"),))
    assert isinstance(result, Err)
    assert not out.exists()
    meta_diff = ArtifactRef(
        path=ref_a.path,
        sha256=ref_a.sha256,
        size_bytes=ref_a.size_bytes,
        kind="FIGURE",
        format=ref_a.format,
    )
    split2 = SplitEvidence(
        task="segmentor", split="test", dataset_hash=_HEX_B, artifacts=(meta_diff,)
    )
    result2 = publish_bundle(
        tmp_path / "conflict2",
        _summary(artifacts=(ref_a,), splits=(split2,)),
        (BundleFile(path="t.csv", data=b"a"),),
    )
    assert isinstance(result2, Err)
    assert not (tmp_path / "conflict2").exists()


def test_verify_bundle_rejects_escaped_targets(tmp_path: Path) -> None:
    """A junction/symlink escape inside the bundle fails verification."""
    data = b"inside-ok"
    ref = _ref_for("dir/f.bin", data)
    out = _publish(tmp_path, _summary(artifacts=(ref,)), (BundleFile(path="dir/f.bin", data=data),))
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "f.bin").write_bytes(data)
    shutil.rmtree(out / "dir")
    linked = False
    try:
        os.symlink(outside, out / "dir", target_is_directory=True)
        linked = True
    except OSError:
        proc = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(out / "dir"), str(outside)],
            capture_output=True,
        )
        linked = proc.returncode == 0
    if not linked:
        pytest.skip("no privilege to create a link for the escape test")
    result = verify_bundle(out)
    assert isinstance(result, Err)


def test_table_schema_and_column_validation() -> None:
    """Schemas require unique named columns with typed dtypes."""
    schema = TableSchema(
        columns=(
            ColumnSpec(name="tile", dtype="STRING"),
            ColumnSpec(name="n", dtype="INT64", nullable=True),
        )
    )
    assert len(schema.columns) == 2
    with pytest.raises(ValueError):
        TableSchema(columns=())
    with pytest.raises(ValueError):
        TableSchema(
            columns=(
                ColumnSpec(name="x", dtype="STRING"),
                ColumnSpec(name="x", dtype="INT64"),
            )
        )
    with pytest.raises(ValueError):
        ColumnSpec(name="", dtype="STRING")
    with pytest.raises(ValueError):
        ColumnSpec(name="x", dtype="TEXT")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ColumnSpec(name="x", dtype="STRING", nullable=1)  # type: ignore[arg-type]


def _mixed_schema() -> TableSchema:
    return TableSchema(
        columns=(
            ColumnSpec(name="name", dtype="STRING"),
            ColumnSpec(name="count", dtype="INT64", nullable=True),
            ColumnSpec(name="score", dtype="FLOAT64"),
            ColumnSpec(name="flag", dtype="BOOLEAN", nullable=True),
        )
    )


def test_csv_table_roundtrip(tmp_path: Path) -> None:
    """Typed CSV preserves nulls, escapes, commas, newlines, and precision."""
    schema = _mixed_schema()
    rows: tuple[dict[str, Scalar], ...] = (
        {"name": "plain", "count": 3, "score": 0.1 + 0.2, "flag": True},
        {"name": "", "count": None, "score": 1e300, "flag": None},
        {"name": "\\N", "count": -5, "score": -0.0, "flag": False},
        {"name": "\\literal", "count": 0, "score": 1.0, "flag": True},
        {"name": 'with,comma\nand"quote"', "count": 7, "score": 2.5, "flag": False},
    )
    path = tmp_path / "t.csv"
    ref = write_table(path, rows, schema)
    assert isinstance(ref, Ok)
    assert ref.value.format == "csv"
    assert ref.value.kind == "TABLE"
    assert ref.value.rows == len(rows)
    assert ref.value.path == "t.csv"
    read = read_table(path, schema)
    assert isinstance(read, Ok)
    assert read.value == rows
    text = path.read_text(encoding="utf-8")
    assert "\\\\N" in text


def test_csv_table_empty_preserves_header(tmp_path: Path) -> None:
    """Empty tables still carry their typed header."""
    schema = _mixed_schema()
    path = tmp_path / "empty.csv"
    assert isinstance(write_table(path, (), schema), Ok)
    read = read_table(path, schema)
    assert isinstance(read, Ok)
    assert read.value == ()
    header = path.read_text(encoding="utf-8").splitlines()[0]
    assert set(header.split(",")) == {"name", "count", "score", "flag"}


def test_table_write_rejects_bad_rows(tmp_path: Path) -> None:
    """Row keys and scalar types are checked exactly against the schema."""
    schema = _mixed_schema()
    path = tmp_path / "t.csv"
    missing = write_table(path, ({"name": "x"},), schema)
    assert isinstance(missing, Err)
    assert not path.exists()
    extra = write_table(
        path,
        ({"name": "x", "count": 1, "score": 0.1, "flag": True, "surprise": 1},),
        schema,
    )
    assert isinstance(extra, Err)
    bool_count = write_table(
        path,
        ({"name": "x", "count": True, "score": 0.1, "flag": False},),
        schema,
    )
    assert isinstance(bool_count, Err)
    big = write_table(
        path,
        ({"name": "x", "count": 2**63, "score": 0.1, "flag": False},),
        schema,
    )
    assert isinstance(big, Err)
    nan = write_table(
        path,
        ({"name": "x", "count": 1, "score": float("nan"), "flag": True},),
        schema,
    )
    assert isinstance(nan, Err)
    not_str = write_table(
        path,
        ({"name": 3, "count": 1, "score": 0.1, "flag": True},),
        schema,
    )
    assert isinstance(not_str, Err)
    int_as_flag = write_table(
        path,
        ({"name": "x", "count": 1, "score": 0.1, "flag": 1},),
        schema,
    )
    assert isinstance(int_as_flag, Err)
    huge = write_table(
        path,
        ({"name": "x", "count": 1, "score": 10**400, "flag": True},),
        schema,
    )
    assert isinstance(huge, Err)
    assert not path.exists()


def test_table_float64_int_normalization(tmp_path: Path) -> None:
    """Finite ints convert to float identically in CSV and Parquet."""
    schema = _mixed_schema()
    rows: tuple[dict[str, Scalar], ...] = (
        {"name": "x", "count": 1, "score": 3, "flag": True},
        {"name": "y", "count": 2, "score": 2**100, "flag": False},
    )
    expected: tuple[dict[str, Scalar], ...] = (
        {"name": "x", "count": 1, "score": 3.0, "flag": True},
        {"name": "y", "count": 2, "score": float(2**100), "flag": False},
    )
    for suffix in (".csv", ".parquet"):
        path = tmp_path / f"t{suffix}"
        assert isinstance(write_table(path, rows, schema), Ok)
        read = read_table(path, schema)
        assert isinstance(read, Ok)
        assert read.value == expected
    for suffix in (".csv", ".parquet"):
        path = tmp_path / f"big{suffix}"
        result = write_table(
            path,
            ({"name": "x", "count": 1, "score": 10**400, "flag": True},),
            schema,
        )
        assert isinstance(result, Err)
        assert not path.exists()


def test_table_write_rejects_null_in_nonnullable(tmp_path: Path) -> None:
    """None in a non-nullable column fails; supported suffixes only."""
    schema = _mixed_schema()
    path = tmp_path / "t.csv"
    null_name = write_table(
        path,
        ({"name": None, "count": 1, "score": 0.1, "flag": True},),
        schema,
    )
    assert isinstance(null_name, Err)
    unsupported = write_table(
        tmp_path / "t.json",
        ({"name": "x", "count": 1, "score": 0.1, "flag": True},),
        schema,
    )
    assert isinstance(unsupported, Err)


def test_table_write_is_exclusive(tmp_path: Path) -> None:
    """write_table refuses to overwrite an existing file."""
    schema = _mixed_schema()
    path = tmp_path / "t.csv"
    row: dict[str, Scalar] = {"name": "x", "count": 1, "score": 0.1, "flag": True}
    assert isinstance(write_table(path, (row,), schema), Ok)
    assert isinstance(write_table(path, (row,), schema), Err)
    assert path.read_text(encoding="utf-8").count("\n") == 2


def test_csv_read_wrong_schema_and_nullability(tmp_path: Path) -> None:
    """The reader honors the supplied schema and null markers exactly."""
    schema = _mixed_schema()
    path = tmp_path / "t.csv"
    path.write_text("name,count,score,flag\nx,1,0.5,true\n", encoding="utf-8")
    wrong = TableSchema(columns=(ColumnSpec(name="name", dtype="STRING"),))
    assert isinstance(read_table(path, wrong), Err)
    null_path = tmp_path / "null.csv"
    null_path.write_text("name,count,score,flag\n\\N,1,0.5,true\n", encoding="utf-8")
    assert isinstance(read_table(null_path, schema), Err)
    ok_path = tmp_path / "ok.csv"
    ok_path.write_text("name,count,score,flag\n,\\N,1e300,\\N\n", encoding="utf-8")
    read = read_table(ok_path, schema)
    assert isinstance(read, Ok)
    assert read.value[0]["name"] == ""
    assert read.value[0]["count"] is None
    assert read.value[0]["flag"] is None
    bad_int = tmp_path / "bad.csv"
    bad_int.write_text("name,count,score,flag\nx,1.5,0.5,true\n", encoding="utf-8")
    assert isinstance(read_table(bad_int, schema), Err)


def test_csv_read_rejects_noncanonical_input(tmp_path: Path) -> None:
    """Single leading slashes, bad UTF-8, and unclosed quotes are errors."""
    schema = _mixed_schema()
    bad_escape = tmp_path / "escape.csv"
    bad_escape.write_text("name,count,score,flag\n\\single,1,0.5,true\n", encoding="utf-8")
    assert isinstance(read_table(bad_escape, schema), Err)
    bad_utf8 = tmp_path / "utf8.csv"
    bad_utf8.write_bytes(b"name,count,score,flag\n\xff,1,0.5,true\n")
    assert isinstance(read_table(bad_utf8, schema), Err)
    unclosed = tmp_path / "unclosed.csv"
    unclosed.write_text('name,count,score,flag\n"x,1,0.5,true\n', encoding="utf-8")
    assert isinstance(read_table(unclosed, schema), Err)
    doubled = tmp_path / "doubled.csv"
    doubled.write_text("name,count,score,flag\n\\\\single,1,0.5,true\n", encoding="utf-8")
    read = read_table(doubled, schema)
    assert isinstance(read, Ok)
    assert read.value[0]["name"] == "\\single"


def test_parquet_table_roundtrip(tmp_path: Path) -> None:
    """Parquet tables round-trip with explicit schema and nulls."""
    schema = _mixed_schema()
    rows: tuple[dict[str, Scalar], ...] = (
        {"name": "a", "count": 1, "score": 0.1, "flag": True},
        {"name": "", "count": None, "score": 1e300, "flag": None},
        {"name": "\\N", "count": -7, "score": 0.0, "flag": False},
    )
    path = tmp_path / "t.parquet"
    ref = write_table(path, rows, schema, bundle_path="tables/t.parquet")
    assert isinstance(ref, Ok)
    assert ref.value.path == "tables/t.parquet"
    assert ref.value.format == "parquet"
    read = read_table(path, schema)
    assert isinstance(read, Ok)
    assert read.value == rows


def test_parquet_empty_and_schema_mismatch(tmp_path: Path) -> None:
    """Empty parquet preserves types; a wrong schema is rejected."""
    schema = _mixed_schema()
    path = tmp_path / "empty.parquet"
    assert isinstance(write_table(path, (), schema), Ok)
    read = read_table(path, schema)
    assert isinstance(read, Ok)
    assert read.value == ()
    wrong = TableSchema(
        columns=(
            ColumnSpec(name="name", dtype="STRING"),
            ColumnSpec(name="count", dtype="STRING", nullable=True),
            ColumnSpec(name="score", dtype="FLOAT64"),
            ColumnSpec(name="flag", dtype="BOOLEAN", nullable=True),
        )
    )
    assert isinstance(read_table(path, wrong), Err)


def test_read_table_rejects_unsupported_suffix(tmp_path: Path) -> None:
    """Only explicit .csv/.parquet inputs are read."""
    schema = _mixed_schema()
    path = tmp_path / "t.json"
    path.write_text("[]", encoding="utf-8")
    assert isinstance(read_table(path, schema), Err)
