"""Frozen generalization evidence serializes into checksummed bundle artifacts."""

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import tools.ml_models.analysis.generalization_artifacts as artifact_module
from flight.libs.types import Err, Ok, Result
from tools.ml_models.analysis.artifacts import Scalar, read_table
from tools.ml_models.analysis.capture import CaptureRow
from tools.ml_models.analysis.config import GeneralizationConfig, ScoreConfig
from tools.ml_models.analysis.contracts import (
    ArtifactRef,
    MetricSupport,
    MetricValue,
    SampleKey,
    Split,
)
from tools.ml_models.analysis.generalization_artifacts import (
    GENERALIZATION_METRICS_SCHEMA,
    INTERVAL_AUDIT_SCHEMA,
    GeneralizationArtifacts,
    generalization_artifacts,
)
from tools.ml_models.analysis.metrics.generalization import (
    GeneralizationEvidence,
    development_gaps,
    fit_baseline,
    measure_generalization,
)


def _row(index: int, group: str, label: int, logit: float, *, split: Split = "val") -> CaptureRow:
    return CaptureRow(
        key=SampleKey(
            dataset_hash="d" * 64,
            task="classifier",
            split=split,
            spatial_shard=(2, 3),
            row_index=index,
            tile_id=f"tile-{index}",
            element="id",
        ),
        group_id=group,
        bin_id="",
        label=label,
        gsd_m=(2.0, 3.0),
        metrics=(
            MetricValue(
                name="logit",
                value=logit,
                status="AVAILABLE",
                support=MetricSupport(unit="IMAGE", n=1),
            ),
        ),
        failure_score=0,
    )


def _canonical(data: object) -> bytes:
    return json.dumps(data, sort_keys=True, allow_nan=False, separators=(",", ":")).encode("utf-8")


def _two_group_rows() -> tuple[CaptureRow, ...]:
    return (
        _row(0, "a", 1, -4.0),
        _row(1, "a", 1, 4.0),
        _row(2, "b", 0, -4.0),
    )


def _evidence() -> GeneralizationEvidence:
    measured = measure_generalization(
        _two_group_rows(), ScoreConfig(), GeneralizationConfig(bootstrap_replicates=8)
    )
    assert isinstance(measured, Ok)
    return measured.value


def _files(result: Result[GeneralizationArtifacts, str]) -> dict[str, bytes]:
    assert isinstance(result, Ok)
    return {file.path: file.data for file in result.value.files}


def _metric_row(
    metric: MetricValue, population: str, name: str | None, value: str | None
) -> dict[str, Scalar]:
    interval = metric.interval
    return {
        "population": population,
        "stratum_name": name,
        "stratum_value": value,
        "name": metric.name,
        "value": float(metric.value) if metric.value is not None else None,
        "status": metric.status,
        "reason": metric.reason,
        "unit": metric.unit,
        "aggregation": metric.aggregation,
        "support_unit": metric.support.unit,
        "support_n": metric.support.n,
        "threshold": float(metric.threshold) if metric.threshold is not None else None,
        "interval_lower": interval.lower if interval is not None else None,
        "interval_upper": interval.upper if interval is not None else None,
        "interval_confidence": interval.confidence if interval is not None else None,
        "interval_method": interval.method if interval is not None else None,
        "interval_replicates": interval.n_replicates if interval is not None else None,
        "interval_valid": interval.n_valid if interval is not None else None,
        "interval_seed": interval.seed if interval is not None else None,
    }


def test_artifacts_bind_every_file_to_a_checksummed_reference() -> None:
    evidence = _evidence()
    result = generalization_artifacts(evidence)
    assert isinstance(result, Ok)
    files = {file.path: file.data for file in result.value.files}
    refs = {ref.path: ref for ref in result.value.refs}
    assert (
        files.keys()
        == refs.keys()
        == {
            "generalization.json",
            "metric-reference.json",
            "tables/schema.json",
            "tables/generalization_metrics.parquet",
            "tables/interval_audit.csv",
        }
    )
    assert files["generalization.json"] == _canonical(asdict(evidence))
    for path, ref in refs.items():
        assert ref.sha256 == hashlib.sha256(files[path]).hexdigest()
        assert ref.size_bytes == len(files[path])
    schemas = json.loads(files["tables/schema.json"])
    assert schemas == {
        "tables/generalization_metrics.parquet": json.loads(
            _canonical(asdict(GENERALIZATION_METRICS_SCHEMA))
        ),
        "tables/interval_audit.csv": json.loads(_canonical(asdict(INTERVAL_AUDIT_SCHEMA))),
    }
    n_metrics = (
        len(evidence.metrics)
        + sum(len(stratum.metrics) for stratum in evidence.strata)
        + len(evidence.baseline_metrics)
    )
    assert refs["tables/generalization_metrics.parquet"].rows == n_metrics
    n_audits = len(evidence.intervals) + sum(
        len(stratum.intervals) for stratum in evidence.stratum_intervals
    )
    assert refs["tables/interval_audit.csv"].rows == n_audits


def test_metric_table_rows_carry_frozen_values_intervals_and_nulls(
    tmp_path: Path,
) -> None:
    evidence = _evidence()
    files = _files(generalization_artifacts(evidence))
    target = tmp_path / "metrics.parquet"
    target.write_bytes(files["tables/generalization_metrics.parquet"])
    parsed = read_table(target, GENERALIZATION_METRICS_SCHEMA)
    assert isinstance(parsed, Ok)
    expected = [_metric_row(metric, "cohort", None, None) for metric in evidence.metrics]
    for stratum in evidence.strata:
        expected += [
            _metric_row(metric, "stratum", stratum.name, stratum.value)
            for metric in stratum.metrics
        ]
    expected += [
        _metric_row(metric, "baseline", None, None) for metric in evidence.baseline_metrics
    ]
    assert parsed.value == tuple(expected)
    assert any(row["interval_lower"] is not None for row in parsed.value)


def test_interval_audit_is_verbatim_and_ineligible_counts_attempted_zero(
    tmp_path: Path,
) -> None:
    evidence = _evidence()
    single = measure_generalization(
        (_row(0, "only", 1, 2.0),), ScoreConfig(), GeneralizationConfig(bootstrap_replicates=8)
    )
    assert isinstance(single, Ok)
    files = _files(generalization_artifacts(single.value))
    target = tmp_path / "audit.csv"
    target.write_bytes(files["tables/interval_audit.csv"])
    parsed = read_table(target, INTERVAL_AUDIT_SCHEMA)
    assert isinstance(parsed, Ok)
    expected = [
        {
            "population": "cohort",
            "stratum_name": None,
            "stratum_value": None,
            "metric": audit.metric,
            "n_replicates": audit.n_replicates,
            "n_attempted": audit.n_attempted,
            "n_valid": audit.n_valid,
            "n_invalid": audit.n_invalid,
            "n_groups": audit.n_groups,
            "reason": audit.reason,
        }
        for audit in single.value.intervals
    ]
    for stratum in single.value.stratum_intervals:
        expected += [
            {
                "population": "stratum",
                "stratum_name": stratum.name,
                "stratum_value": stratum.value,
                "metric": audit.metric,
                "n_replicates": audit.n_replicates,
                "n_attempted": audit.n_attempted,
                "n_valid": audit.n_valid,
                "n_invalid": audit.n_invalid,
                "n_groups": audit.n_groups,
                "reason": audit.reason,
            }
            for audit in stratum.intervals
        ]
    assert parsed.value == tuple(expected)
    assert all(row["n_attempted"] == 0 for row in parsed.value)
    assert all(row["reason"] for row in parsed.value)
    assert len(evidence.intervals) > 0


def test_metric_reference_keeps_unknown_names_explicit() -> None:
    evidence = _evidence()
    unknown = MetricValue(
        name="unlisted_diagnostic",
        value=None,
        status="UNAVAILABLE",
        reason="not captured",
        support=MetricSupport(unit="IMAGE", n=1),
    )
    enriched = replace(evidence, metrics=evidence.metrics + (unknown,))
    files = _files(generalization_artifacts(enriched))
    reference = json.loads(files["metric-reference.json"])
    names = {
        metric.name
        for metric in (
            *enriched.metrics,
            *enriched.baseline_metrics,
            *(m for s in enriched.strata for m in s.metrics),
        )
    }
    assert {entry["name"] for entry in reference} == names
    entry = next(e for e in reference if e["name"] == "unlisted_diagnostic")
    assert entry["definition"] is None and entry["reason"]
    known = next(e for e in reference if e["name"] == "accuracy")
    assert known["definition"] is not None and known["reason"] is None


def test_development_gaps_bind_as_optional_companion() -> None:
    train = (_row(0, "a", 1, 4.0, split="train"), _row(1, "b", 0, -4.0, split="train"))
    validation = (_row(0, "c", 1, 4.0), _row(1, "d", 0, -4.0))
    fitted = fit_baseline(train)
    gaps = development_gaps(train, validation, ScoreConfig(), checkpoint_hash="c" * 64)
    assert isinstance(fitted, Ok) and isinstance(gaps, Ok)
    measured = measure_generalization(
        validation,
        ScoreConfig(),
        GeneralizationConfig(bootstrap_replicates=8),
        baseline=fitted.value,
    )
    assert isinstance(measured, Ok)
    without = _files(generalization_artifacts(measured.value))
    assert "development.json" not in without
    files = _files(generalization_artifacts(measured.value, development=gaps.value))
    assert files["development.json"] == _canonical(asdict(gaps.value))
    reference = json.loads(files["metric-reference.json"])
    gap_names = {gap.name for gap in gaps.value.gaps}
    assert gap_names <= {entry["name"] for entry in reference}
    baseline_rows = json.loads(files["generalization.json"])["baseline_metrics"]
    assert baseline_rows


def test_prefix_namespaces_identical_bytes_and_disjoint_refs() -> None:
    evidence = _evidence()
    plain = generalization_artifacts(evidence)
    train = generalization_artifacts(evidence, prefix="evidence/train")
    val = generalization_artifacts(evidence, prefix="evidence/val")
    assert isinstance(plain, Ok) and isinstance(train, Ok) and isinstance(val, Ok)
    plain_files = {file.path: file.data for file in plain.value.files}
    train_files = {file.path: file.data for file in train.value.files}
    val_files = {file.path: file.data for file in val.value.files}
    assert all(path.startswith("evidence/train/") for path in train_files)
    assert all(path.startswith("evidence/val/") for path in val_files)
    assert {p.removeprefix("evidence/train/"): b for p, b in train_files.items()} == plain_files
    assert {p.removeprefix("evidence/val/"): b for p, b in val_files.items()} == plain_files
    plain_refs = {ref.path: ref for ref in plain.value.refs}
    train_refs = {ref.path.removeprefix("evidence/train/"): ref for ref in train.value.refs}
    val_refs = {ref.path.removeprefix("evidence/val/"): ref for ref in val.value.refs}
    assert train_refs.keys() == val_refs.keys() == plain_refs.keys()
    for path, ref in plain_refs.items():
        assert train_refs[path].sha256 == ref.sha256
        assert train_refs[path].rows == ref.rows
        assert train_refs[path].path == f"evidence/train/{path}"


@pytest.mark.parametrize("prefix", ["../escape", "/absolute", r"back\slash", "a//b", "trail. "])
def test_prefix_rejects_unsafe_namespaces(prefix: str) -> None:
    assert isinstance(generalization_artifacts(_evidence(), prefix=prefix), Err)


def test_development_companion_must_share_evidence_identity() -> None:
    train = (
        _row(0, "a", 1, 4.0, split="train"),
        _row(1, "b", 0, -4.0, split="train"),
    )
    validation = (_row(0, "c", 1, 4.0), _row(1, "d", 0, -4.0))
    gaps = development_gaps(train, validation, ScoreConfig(), checkpoint_hash="c" * 64)
    measured = measure_generalization(
        validation, ScoreConfig(), GeneralizationConfig(bootstrap_replicates=8)
    )
    assert isinstance(gaps, Ok) and isinstance(measured, Ok)
    assert isinstance(
        generalization_artifacts(
            replace(measured.value, task="segmentor"),
            development=gaps.value,
        ),
        Err,
    )
    assert isinstance(generalization_artifacts(measured.value, development=gaps.value), Ok)
    assert isinstance(
        generalization_artifacts(
            measured.value, development=replace(gaps.value, dataset_hash="1" * 64)
        ),
        Err,
    )
    assert isinstance(
        generalization_artifacts(
            measured.value,
            development=replace(gaps.value, dataset_manifest_hash="2" * 64),
        ),
        Err,
    )
    assert isinstance(
        generalization_artifacts(
            measured.value,
            development=replace(
                gaps.value,
                score_config=ScoreConfig(classifier_probability_threshold=0.6),
            ),
        ),
        Err,
    )


def _forced_table_failure(*args: object, **kwargs: object) -> Result[ArtifactRef, str]:
    return Err("forced codec failure")


def test_table_codec_failure_returns_err(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(artifact_module, "write_table", _forced_table_failure)
    result = generalization_artifacts(_evidence())
    assert isinstance(result, Err)
    assert "forced codec failure" in result.error


def test_nonfinite_companion_value_fails_canonical_json() -> None:
    train = (
        _row(0, "a", 1, 4.0, split="train"),
        _row(1, "b", 0, -4.0, split="train"),
    )
    validation = (_row(0, "c", 1, 4.0), _row(1, "d", 0, -4.0))
    gaps = development_gaps(train, validation, ScoreConfig(), checkpoint_hash="c" * 64)
    measured = measure_generalization(
        validation, ScoreConfig(), GeneralizationConfig(bootstrap_replicates=8)
    )
    assert isinstance(gaps, Ok) and isinstance(measured, Ok)
    assert gaps.value.gaps
    broken = replace(gaps.value, gaps=(replace(gaps.value.gaps[0], difference=float("nan")),))
    assert isinstance(generalization_artifacts(measured.value, development=broken), Err)
