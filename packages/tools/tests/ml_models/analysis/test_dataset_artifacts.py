"""Tests for dataset evidence publication."""

import json
import subprocess
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
from flight.libs.types import Err, Ok, Result
from tools.ml_models.analysis import dataset_artifacts
from tools.ml_models.analysis.artifacts import (
    BundleFile,
    publish_bundle,
    read_table,
    verify_bundle,
)
from tools.ml_models.analysis.config import DatasetAnalysisConfig
from tools.ml_models.analysis.dataset import analyze_dataset, measure_dataset
from tools.ml_models.analysis.dataset_artifacts import (
    COMPONENTS_SCHEMA,
    COVERAGE_SCHEMA,
    SAMPLES_SCHEMA,
    TABLE_SCHEMAS,
    code_identity,
)
from tools.ml_models.analysis.summaries import DatasetSummary, Summary
from tools.ml_models.dataset.manifest import compute_dataset_hash


def _analyze(dataset: Path, out: Path) -> Result[Path, str]:
    return analyze_dataset(DatasetAnalysisConfig(dataset=str(dataset), out=str(out)))


def test_analyze_publishes_verifiable_measurement_bundle(
    tmp_path: Path, build_synthetic_dataset: Callable[..., Path]
) -> None:
    """A valid dataset publishes one complete, verifiable evidence bundle."""
    dataset = build_synthetic_dataset(tmp_path / "ds")
    measured = measure_dataset(dataset)
    assert isinstance(measured, Ok)
    frozen = measured.value
    out = tmp_path / "analysis"
    result = _analyze(dataset, out)
    assert isinstance(result, Ok)
    assert result.value == out
    assert (out / "summary.json").is_file()
    assert (out / "config.toml").is_file()
    verified = verify_bundle(out)
    assert isinstance(verified, Ok)
    summary = verified.value
    assert isinstance(summary, DatasetSummary)
    assert summary.summary_kind == "DATASET_ANALYSIS"
    assert summary.status == "COMPLETE"
    assert summary.dataset == frozen.identity
    assert summary.metrics == frozen.metrics
    assert summary.splits == frozen.splits
    measurements = json.loads((out / "measurements.json").read_bytes())
    assert measurements == json.loads(
        json.dumps(asdict(frozen), sort_keys=True, allow_nan=False, separators=(",", ":"))
    )
    assert measurements["pixels"]
    assert measurements["coverage"]
    assert measurements["samples"]
    refs = {ref.path: ref for ref in summary.artifacts}
    assert set(refs) == {
        "config.toml",
        "measurements.json",
        "tables/samples.parquet",
        "tables/components.parquet",
        "tables/coverage.csv",
        "tables/schema.json",
    }
    assert refs["tables/samples.parquet"].rows == len(frozen.samples)
    assert refs["tables/components.parquet"].rows == len(frozen.components)
    assert refs["tables/coverage.csv"].rows == len(frozen.coverage)
    for path, ref in refs.items():
        assert ref.path == path
        assert ref.sha256 and ref.size_bytes >= 0
    schemas = json.loads((out / "tables" / "schema.json").read_bytes())
    assert schemas == json.loads(
        json.dumps(
            {rel: asdict(TABLE_SCHEMAS[rel]) for rel in sorted(TABLE_SCHEMAS)},
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        )
    )


def test_published_tables_match_frozen_measurements(
    tmp_path: Path, build_synthetic_dataset: Callable[..., Path]
) -> None:
    """Table rows mirror the frozen samples, components and coverage exactly."""
    dataset = build_synthetic_dataset(tmp_path / "ds")
    measured = measure_dataset(dataset)
    assert isinstance(measured, Ok)
    frozen = measured.value
    out = tmp_path / "analysis"
    assert isinstance(_analyze(dataset, out), Ok)
    samples = read_table(out / "tables" / "samples.parquet", SAMPLES_SCHEMA)
    assert isinstance(samples, Ok)
    assert len(samples.value) == len(frozen.samples)
    for row, sample in zip(samples.value, frozen.samples, strict=True):
        assert row["variant_id"] == sample.variant_id
        assert row["dataset_hash"] == sample.key.dataset_hash
        assert row["task"] == sample.key.task
        assert row["split"] == sample.key.split
        assert row["tile_id"] == sample.row.tile_id
        assert row["element"] == sample.row.element
        assert row["group_id"] == sample.row.group_id
        assert row["label"] == sample.label
        assert row["lateral_gsd_m"] == sample.gsd_m[0]
        assert row["along_gsd_m"] == sample.gsd_m[1]
        assert row["gsd_nominal"] == sample.row.gsd_nominal
        assert row["stored_rows"] == sample.stored_rows
        assert row["image_sha256"] == sample.image_sha256
        assert row["observation_id"] == sample.row.metadata.observation_id
        assert row["acquired_at_utc"] == sample.row.metadata.acquired_at_utc
        assert row["source_annotation_state"] == sample.row.metadata.source_annotation_state
        assert row["prepared_mask_state"] == sample.row.prepared_mask_state
        assert row["mask_area_px"] == (sample.mask.area_px if sample.mask is not None else None)
        assert row["mask_components"] == (
            sample.mask.n_components if sample.mask is not None else None
        )
        assert row["mask_border_touching"] == (
            sample.mask.border_touching if sample.mask is not None else None
        )
        assert row["mask_task"] == (sample.mask_key.task if sample.mask_key is not None else None)
        assert row["mask_row_index"] == (
            sample.mask_key.row_index if sample.mask_key is not None else None
        )
        assert row["mask_element"] == (
            sample.mask_key.element if sample.mask_key is not None else None
        )
    components = read_table(out / "tables" / "components.parquet", COMPONENTS_SCHEMA)
    assert isinstance(components, Ok)
    assert components.value == tuple(
        {
            "variant_id": component.variant_id,
            "component_index": component.component_index,
            "area_px": component.area_px,
            "area_m2": component.area_m2,
        }
        for component in frozen.components
    )
    coverage = read_table(out / "tables" / "coverage.csv", COVERAGE_SCHEMA)
    assert isinstance(coverage, Ok)
    assert coverage.value == tuple(
        {
            "population": record.population,
            "split": record.split,
            "field": record.field,
            "value": record.value,
            "n": record.n,
            "total": record.total,
        }
        for record in frozen.coverage
    )


def test_missing_and_tampered_dataset_refuse_without_output(
    tmp_path: Path, build_synthetic_dataset: Callable[..., Path]
) -> None:
    """Absent or corrupt datasets fail closed and create no bundle."""
    missing = _analyze(tmp_path / "absent", tmp_path / "out-missing")
    assert isinstance(missing, Err)
    assert not (tmp_path / "out-missing").exists()
    dataset = build_synthetic_dataset(tmp_path / "ds")
    (dataset / "injected.txt").write_text("tampered", encoding="utf-8")
    tampered = _analyze(dataset, tmp_path / "out-tampered")
    assert isinstance(tampered, Err)
    assert not (tmp_path / "out-tampered").exists()


def test_output_inside_dataset_and_existing_destination_are_refused(
    tmp_path: Path, build_synthetic_dataset: Callable[..., Path]
) -> None:
    """Outputs inside the dataset or over existing files are refused unchanged."""
    dataset = build_synthetic_dataset(tmp_path / "ds")
    before = compute_dataset_hash(dataset)
    inside = _analyze(dataset, dataset / "analysis")
    assert isinstance(inside, Err)
    assert compute_dataset_hash(dataset) == before
    out = tmp_path / "taken"
    out.mkdir()
    sentinel = out / "sentinel.txt"
    sentinel.write_bytes(b"user bytes")
    existing = _analyze(dataset, out)
    assert isinstance(existing, Err)
    assert sentinel.read_bytes() == b"user bytes"


def test_raced_destination_preserves_user_bytes(
    tmp_path: Path,
    build_synthetic_dataset: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A destination appearing after preflight cannot be overwritten."""
    dataset = build_synthetic_dataset(tmp_path / "ds")
    out = tmp_path / "raced"

    def racer(
        target: Path,
        summary: Summary,
        files: tuple[BundleFile, ...],
        dataset_root: Path | None = None,
    ) -> Result[Path, str]:
        target.mkdir()
        (target / "sentinel.txt").write_bytes(b"concurrent bytes")
        return publish_bundle(target, summary, files, dataset_root=dataset_root)

    monkeypatch.setattr(dataset_artifacts, "publish_bundle", racer)
    result = _analyze(dataset, out)
    assert isinstance(result, Err)
    assert (out / "sentinel.txt").read_bytes() == b"concurrent bytes"


def test_codec_failure_creates_no_output(
    tmp_path: Path,
    build_synthetic_dataset: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A table codec error aborts before the destination is reserved."""
    dataset = build_synthetic_dataset(tmp_path / "ds")
    out = tmp_path / "out"

    def broken_codec(*args: object, **kwargs: object) -> Result[object, str]:
        return Err("simulated codec failure")

    monkeypatch.setattr(dataset_artifacts, "write_table", broken_codec)
    result = _analyze(dataset, out)
    assert isinstance(result, Err)
    assert "codec" in result.error
    assert not out.exists()


def test_publication_write_failure_retains_incomplete_marker(
    tmp_path: Path,
    build_synthetic_dataset: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mid-publish write failure leaves the .incomplete marker in place."""
    dataset = build_synthetic_dataset(tmp_path / "ds")
    out = tmp_path / "out"

    def failing(
        self: Path,
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> object:
        if str(self).startswith(str(out)):
            raise OSError("simulated write failure")
        return open(self, mode, buffering, encoding, errors, newline)

    monkeypatch.setattr(Path, "open", failing)
    result = _analyze(dataset, out)
    assert isinstance(result, Err)
    assert (out / ".incomplete").is_file()
    assert isinstance(verify_bundle(out), Err)


def test_preflight_path_failures_return_err_without_output(
    tmp_path: Path,
    build_synthetic_dataset: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unresolvable output paths fail recoverably before any publication."""
    dataset = build_synthetic_dataset(tmp_path / "ds")
    null = _analyze(dataset, Path("bad\0name"))
    assert isinstance(null, Err)

    def boom(self: Path, strict: bool = False) -> Path:
        raise OSError("simulated resolve failure")

    out = tmp_path / "out-boom"
    monkeypatch.setattr(Path, "resolve", boom)
    result = _analyze(dataset, out)
    assert isinstance(result, Err)
    assert "preflight" in result.error
    assert not out.exists()


def test_code_identity_reports_source_worktree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Git commands run in the source module's directory and populate state."""
    calls: list[tuple[tuple[str, ...], object]] = []

    def fake(
        cmd: tuple[str, ...],
        *,
        cwd: object,
        check: object,
        capture_output: object,
        text: object,
        timeout: object,
    ) -> SimpleNamespace:
        calls.append((cmd, cwd))
        if "rev-parse" in cmd:
            return SimpleNamespace(stdout="deadbeef\n")
        return SimpleNamespace(stdout=" M tracked.py\n")

    monkeypatch.setattr(subprocess, "run", fake)
    code = code_identity()
    assert code.revision == "deadbeef"
    assert code.dirty is True
    assert code.diff_hash is None
    assert calls[0][1] == Path(dataset_artifacts.__file__).resolve().parent
    assert calls[1][1] == Path(dataset_artifacts.__file__).resolve().parent


def test_code_identity_unavailable_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing git subprocess yields an explicit unknown identity."""

    def boom(cmd: tuple[str, ...], **kwargs: object) -> SimpleNamespace:
        raise subprocess.CalledProcessError(128, cmd)

    monkeypatch.setattr(subprocess, "run", boom)
    code = code_identity()
    assert code.revision is None
    assert code.dirty is None
    assert code.diff_hash is None
    assert code.reason is not None and code.reason.strip()
