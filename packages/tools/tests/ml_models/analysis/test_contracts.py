"""Tests for the strict typed evidence records in analysis.contracts."""

import dataclasses

import pytest
from tools.ml_models.analysis.contracts import (
    ArtifactRef,
    AvailabilityRecord,
    CheckpointIdentity,
    CodeIdentity,
    ConfidenceInterval,
    CurveEvidence,
    DatasetIdentity,
    MetricSupport,
    MetricValue,
    NamedCount,
    SampleKey,
    SplitEvidence,
    StratumEvidence,
)

_HEX = "a" * 64
_HEX2 = "b" * 64


def _sample_key(**overrides: object) -> SampleKey:
    values: dict[str, object] = {
        "dataset_hash": _HEX,
        "task": "classifier",
        "split": "val",
        "spatial_shard": (193, 258),
        "row_index": 3,
        "tile_id": "tile-1",
        "element": "id",
    }
    values.update(overrides)
    return SampleKey(**values)  # type: ignore[arg-type]


def _metric(**overrides: object) -> MetricValue:
    values: dict[str, object] = {"name": "accuracy", "value": 0.9, "status": "AVAILABLE"}
    values.update(overrides)
    return MetricValue(**values)  # type: ignore[arg-type]


def _artifact_ref(**overrides: object) -> ArtifactRef:
    values: dict[str, object] = {
        "path": "tables/metrics.csv",
        "sha256": _HEX,
        "size_bytes": 12,
        "kind": "TABLE",
        "format": "csv",
    }
    values.update(overrides)
    return ArtifactRef(**values)  # type: ignore[arg-type]


def _dataset_identity(**overrides: object) -> DatasetIdentity:
    values: dict[str, object] = {
        "content_hash": _HEX,
        "manifest_hash": _HEX2,
        "schema_version": 2,
        "source": "fixture",
        "band_names": ("red", "nir"),
        "gsd_reference_m": 0.05,
    }
    values.update(overrides)
    return DatasetIdentity(**values)  # type: ignore[arg-type]


def _split(**overrides: object) -> SplitEvidence:
    values: dict[str, object] = {"task": "segmentor", "split": "test", "dataset_hash": _HEX}
    values.update(overrides)
    return SplitEvidence(**values)  # type: ignore[arg-type]


def test_sample_key_fields() -> None:
    """SampleKey carries the full row-alignment identity."""
    key = _sample_key()
    names = {field.name for field in dataclasses.fields(key)}
    assert names == {
        "dataset_hash",
        "task",
        "split",
        "spatial_shard",
        "row_index",
        "tile_id",
        "element",
    }
    assert key.spatial_shard == (193, 258)
    with pytest.raises(dataclasses.FrozenInstanceError):
        key.row_index = 4  # type: ignore[misc]


def test_sample_key_validation() -> None:
    """SampleKey rejects bad hashes, dims, indices, and empty identifiers."""
    with pytest.raises(ValueError):
        _sample_key(dataset_hash="A" * 64)
    with pytest.raises(ValueError):
        _sample_key(dataset_hash="a" * 63)
    with pytest.raises(ValueError):
        _sample_key(task="detector")
    with pytest.raises(ValueError):
        _sample_key(split="holdout")
    with pytest.raises(ValueError):
        _sample_key(spatial_shard=(0, 258))
    with pytest.raises(ValueError):
        _sample_key(spatial_shard=(True, 258))
    with pytest.raises(ValueError):
        _sample_key(row_index=-1)
    with pytest.raises(ValueError):
        _sample_key(row_index=True)
    with pytest.raises(ValueError):
        _sample_key(row_index=1.5)
    with pytest.raises(ValueError):
        _sample_key(tile_id="")
    with pytest.raises(ValueError):
        _sample_key(element="   ")


def test_named_count_exact_integers() -> None:
    """NamedCount rejects negative, float, and bool values."""
    assert NamedCount(name="predicted", value=3).value == 3
    with pytest.raises(ValueError):
        NamedCount(name="predicted", value=-1)
    with pytest.raises(ValueError):
        NamedCount(name="predicted", value=True)
    with pytest.raises(ValueError):
        NamedCount(name="predicted", value=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        NamedCount(name="", value=1)


def test_metric_support_counts_unique() -> None:
    """MetricSupport requires a nonnegative count and unique count names."""
    support = MetricSupport(unit="PIXEL", n=4, counts=(NamedCount(name="a", value=1),))
    assert support.counts[0].name == "a"
    with pytest.raises(ValueError):
        MetricSupport(unit="BATCH", n=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        MetricSupport(unit="IMAGE", n=-1)
    with pytest.raises(ValueError):
        MetricSupport(unit="IMAGE", n=True)
    dup = (NamedCount(name="x", value=1), NamedCount(name="x", value=2))
    with pytest.raises(ValueError):
        MetricSupport(unit="IMAGE", n=3, counts=dup)


def test_confidence_interval_bounds() -> None:
    """ConfidenceInterval enforces ordered finite endpoints and valid counts."""
    interval = ConfidenceInterval(lower=0.1, upper=0.9)
    assert interval.confidence == 0.95
    assert interval.n_replicates == 1000
    assert interval.n_valid == 1000
    for kwargs in (
        {"lower": 0.9, "upper": 0.1},
        {"lower": float("nan")},
        {"upper": float("inf")},
        {"confidence": 0.0},
        {"confidence": 1.0},
        {"confidence": float("nan")},
        {"confidence": "0.95"},
        {"lower": True},
        {"n_replicates": 0},
        {"n_replicates": True},
        {"n_valid": -1},
        {"n_valid": 1001},
        {"seed": True},
        {"seed": 1.5},
    ):
        bad: dict[str, object] = {"lower": 0.1, "upper": 0.9}
        bad.update(kwargs)
        with pytest.raises(ValueError):
            ConfidenceInterval(**bad)  # type: ignore[arg-type]
    assert ConfidenceInterval(lower=0.5, upper=0.5).lower == 0.5
    assert ConfidenceInterval(lower=0.1, upper=0.9, n_valid=0).n_valid == 0


def test_metric_value_unavailable_reason() -> None:
    """A missing measurement is explicit, never a fabricated number."""
    unavailable = MetricValue(name="roc_auc", value=None, status="UNAVAILABLE", reason="one class")
    assert unavailable.value is None
    assert unavailable.reason == "one class"
    available = MetricValue(name="accuracy", value=0.9, status="AVAILABLE")
    assert available.reason is None
    assert available.unit == "dimensionless"
    assert available.aggregation == "per_image_mean"
    assert available.support == MetricSupport(unit="IMAGE", n=0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        available.value = 0.1  # type: ignore[misc]


def test_metric_value_status_consistency() -> None:
    """Status, value, and reason combinations are validated."""
    with pytest.raises(ValueError):
        _metric(value=None)
    with pytest.raises(ValueError):
        _metric(reason="one class")
    with pytest.raises(ValueError):
        _metric(status="UNAVAILABLE", reason="ok")
    with pytest.raises(ValueError):
        MetricValue(name="x", value=None, status="UNAVAILABLE")
    with pytest.raises(ValueError):
        MetricValue(name="x", value=None, status="UNAVAILABLE", reason="  ")
    with pytest.raises(ValueError):
        _metric(status="SKIP")
    for value in (float("nan"), float("inf"), -float("inf"), True, "0.9"):
        with pytest.raises(ValueError):
            _metric(value=value)
    with pytest.raises(ValueError):
        _metric(threshold=float("nan"))
    with pytest.raises(ValueError):
        _metric(threshold="0.5")
    _metric(threshold=0.5)
    assert _metric(value=3).value == 3.0


def test_metric_value_name_and_interval() -> None:
    """Metric names are lower snake case; intervals need not enclose the value."""
    for name in ("", "AUC", "roc auc", "_roc", "roc_", "roc__auc", "roc-auc"):
        with pytest.raises(ValueError):
            _metric(name=name)
    for name in ("accuracy", "roc_auc", "f1", "iou_50"):
        _metric(name=name)
    inside = ConfidenceInterval(lower=0.8, upper=0.95)
    _metric(interval=inside)
    disjoint = ConfidenceInterval(lower=0.95, upper=0.99)
    assert _metric(interval=disjoint).interval == disjoint
    with pytest.raises(ValueError):
        MetricValue(
            name="x",
            value=None,
            status="UNAVAILABLE",
            reason="n/a",
            interval=inside,
        )


def test_curve_evidence_alignment() -> None:
    """Curve arrays align; histograms declare their approximation."""
    support = MetricSupport(unit="GROUP", n=2)
    curve = CurveEvidence(
        name="reliability",
        x_name="confidence",
        y_name="accuracy",
        x=(0.1, 0.9),
        y=(0.2, 0.8),
        x_unit="probability",
        y_unit="probability",
        support=support,
    )
    assert curve.method == "EXACT"
    assert curve.thresholds == ()
    none_y = CurveEvidence(
        name="c",
        x_name="x",
        y_name="y",
        x=(0.0, 1.0),
        y=(0.5, None),
        x_unit="t",
        y_unit="t",
        support=support,
        thresholds=(0.5, None),
    )
    assert none_y.y == (0.5, None)
    with pytest.raises(ValueError):
        CurveEvidence(
            name="c",
            x_name="x",
            y_name="y",
            x=(0.0,),
            y=(0.0, 1.0),
            x_unit="t",
            y_unit="t",
            support=support,
        )
    with pytest.raises(ValueError):
        CurveEvidence(
            name="c",
            x_name="x",
            y_name="y",
            x=(0.0,),
            y=(0.0,),
            x_unit="t",
            y_unit="t",
            support=support,
            thresholds=(0.5, 0.6),
        )
    with pytest.raises(ValueError):
        CurveEvidence(
            name="c",
            x_name="x",
            y_name="y",
            x=(float("inf"),),
            y=(0.0,),
            x_unit="t",
            y_unit="t",
            support=support,
        )
    with pytest.raises(ValueError):
        CurveEvidence(
            name="c",
            x_name="x",
            y_name="y",
            x=(0.0,),
            y=(0.0,),
            x_unit="t",
            y_unit="t",
            support=support,
            thresholds=(float("inf"),),
        )
    with pytest.raises(ValueError):
        CurveEvidence(
            name="c",
            x_name="x",
            y_name="y",
            x=(True,),
            y=(0.0,),
            x_unit="t",
            y_unit="t",
            support=support,
        )
    with pytest.raises(ValueError):
        CurveEvidence(
            name="c",
            x_name="x",
            y_name="y",
            x=(0.0,),
            y=("0.5",),  # type: ignore[arg-type]
            x_unit="t",
            y_unit="t",
            support=support,
        )


def test_curve_evidence_histogram_contract() -> None:
    """HISTOGRAM curves need n_bins>=2 and a note; EXACT forbids n_bins."""
    support = MetricSupport(unit="GROUP", n=10)
    base: dict[str, object] = dict(
        name="size_hist",
        x_name="size",
        y_name="count",
        x=(0.0, 1.0),
        y=(0.5, 0.5),
        x_unit="px",
        y_unit="n",
        support=support,
    )
    histogram = CurveEvidence(
        **base,  # type: ignore[arg-type]
        method="HISTOGRAM",
        n_bins=4,
        notes=("edges from a 4-bin histogram approximation",),
    )
    assert histogram.method == "HISTOGRAM"
    with pytest.raises(ValueError):
        CurveEvidence(**base, method="HISTOGRAM", n_bins=4)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        CurveEvidence(**base, method="HISTOGRAM", notes=("binned",))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        CurveEvidence(
            **base,  # type: ignore[arg-type]
            method="HISTOGRAM",
            n_bins=1,
            notes=("binned",),
        )
    with pytest.raises(ValueError):
        CurveEvidence(**base, method="EXACT", n_bins=4)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        CurveEvidence(**base, method="BUCKETED")  # type: ignore[arg-type]


def test_artifact_ref_path_safety() -> None:
    """ArtifactRef paths are safe relative POSIX references."""
    assert _artifact_ref().path == "tables/metrics.csv"
    for bad in (
        "",
        "/abs/path.csv",
        "C:/data/x.csv",
        "c:/x.csv",
        "a\\b.csv",
        "../x.csv",
        "a/../x.csv",
        "./x.csv",
        "a/./x.csv",
        "a//x.csv",
        "a/",
        ".",
        "..",
        "a:b.csv",
        "dir/file.csv:stream",
        "x\x00y.csv",
        "x\x7fy.csv",
        "x\ny.csv",
        "dir./x.csv",
        "dir /x.csv",
        "CON.csv",
        "con.csv",
        "lpt1",
        "a/AUX.png",
        "x/NUL.tar",
    ):
        with pytest.raises(ValueError):
            _artifact_ref(path=bad)
    for good in ("a/b/c.csv", "x.parquet", "deep-1/under_score/x-2.png"):
        assert _artifact_ref(path=good).path == good


def test_artifact_ref_fields() -> None:
    """ArtifactRef validates hash, sizes, kind, and format."""
    with pytest.raises(ValueError):
        _artifact_ref(sha256="A" * 64)
    with pytest.raises(ValueError):
        _artifact_ref(sha256="z" * 64)
    with pytest.raises(ValueError):
        _artifact_ref(size_bytes=-1)
    with pytest.raises(ValueError):
        _artifact_ref(size_bytes=True)
    with pytest.raises(ValueError):
        _artifact_ref(rows=-1)
    with pytest.raises(ValueError):
        _artifact_ref(rows=True)
    with pytest.raises(ValueError):
        _artifact_ref(kind="BLOB")
    with pytest.raises(ValueError):
        _artifact_ref(format="")
    with pytest.raises(ValueError):
        _artifact_ref(population="")
    ref = _artifact_ref(rows=10, population="test")
    assert ref.rows == 10


def test_dataset_identity_fields() -> None:
    """DatasetIdentity binds content and manifest hashes to metadata."""
    identity = _dataset_identity()
    assert identity.content_hash == _HEX
    for kwargs in (
        {"content_hash": "x"},
        {"manifest_hash": "B" * 64},
        {"schema_version": 0},
        {"schema_version": True},
        {"source": ""},
        {"source": "   "},
        {"band_names": ()},
        {"band_names": ("red", "red")},
        {"band_names": ("red", "")},
        {"gsd_reference_m": 0.0},
        {"gsd_reference_m": float("nan")},
    ):
        with pytest.raises(ValueError):
            _dataset_identity(**kwargs)


def test_code_identity_requires_reason_when_unknown() -> None:
    """Missing revision or dirty information requires an explicit reason."""
    known = CodeIdentity(revision="abc123", dirty=False)
    assert known.reason is None
    unknown = CodeIdentity(revision=None, dirty=None, reason="git unavailable")
    assert unknown.dirty is None
    with pytest.raises(ValueError):
        CodeIdentity(revision=None, dirty=None)
    with pytest.raises(ValueError):
        CodeIdentity(revision="abc", dirty=None, reason=None)
    with pytest.raises(ValueError):
        CodeIdentity(revision=None, dirty=False, reason="  ")
    with pytest.raises(ValueError):
        CodeIdentity(revision="abc", dirty=False, diff_hash="xyz")
    with pytest.raises(ValueError):
        CodeIdentity(revision="abc", dirty=0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        CodeIdentity(revision="abc", dirty="no")  # type: ignore[arg-type]
    CodeIdentity(revision="abc", dirty=True, diff_hash=_HEX)


def test_checkpoint_identity_fields() -> None:
    """CheckpointIdentity binds a hash, task, arch, and provenance."""
    checkpoint = CheckpointIdentity(
        sha256=_HEX,
        kind="segmentor",
        arch="unet",
        training_dataset_hash=_HEX2,
    )
    assert checkpoint.epoch is None
    known = CheckpointIdentity(
        sha256=_HEX,
        kind="classifier",
        arch="resnet",
        training_dataset_hash=_HEX2,
        epoch=3,
        step=100,
    )
    assert known.step == 100
    for kwargs in (
        {"sha256": "x"},
        {"training_dataset_hash": "y"},
        {"arch": ""},
        {"epoch": -1},
        {"epoch": True},
        {"step": 1.5},
    ):
        bad: dict[str, object] = {
            "sha256": _HEX,
            "kind": "segmentor",
            "arch": "unet",
            "training_dataset_hash": _HEX2,
        }
        bad.update(kwargs)
        with pytest.raises(ValueError):
            CheckpointIdentity(**bad)  # type: ignore[arg-type]


def test_availability_record_reasons() -> None:
    """Unavailable and skipped outputs carry explicit reasons."""
    available = AvailabilityRecord(name="figures", status="AVAILABLE")
    assert available.required is False
    skipped = AvailabilityRecord(name="predictions", status="SKIPPED", reason="compact capture")
    assert skipped.reason == "compact capture"
    with pytest.raises(ValueError):
        AvailabilityRecord(name="x", status="UNAVAILABLE")
    with pytest.raises(ValueError):
        AvailabilityRecord(name="x", status="SKIPPED", reason="")
    with pytest.raises(ValueError):
        AvailabilityRecord(name="x", status="AVAILABLE", reason="why not")
    with pytest.raises(ValueError):
        AvailabilityRecord(name="x", status="PENDING")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        AvailabilityRecord(name="", status="AVAILABLE")
    with pytest.raises(ValueError):
        AvailabilityRecord(name="x", status="AVAILABLE", required=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        AvailabilityRecord(name="x", status="AVAILABLE", required="yes")  # type: ignore[arg-type]


def test_split_evidence_defaults() -> None:
    """SplitEvidence carries task, split, and dataset identity."""
    evidence = _split()
    assert evidence.metrics == ()
    assert evidence.curves == ()
    assert evidence.artifacts == ()
    assert evidence.support == MetricSupport(unit="IMAGE", n=0)
    assert evidence.checkpoint_hash is None
    named = _split(metrics=(MetricValue(name="iou", value=0.5, status="AVAILABLE"),))
    assert named.metrics[0].name == "iou"


def test_stratum_identity_keeps_missing_values_distinct() -> None:
    """Missing bins cannot collide with literal names or colon-containing identifiers."""
    support = MetricSupport(unit="IMAGE", n=1)
    records = tuple(
        StratumEvidence(name=name, value=value, metrics=(), support=support)
        for name, value in (("gsd_bin", None), ("gsd_bin", "None"), ("a:b", "c"), ("a", "b:c"))
    )
    assert _split(strata=records).strata == records
    with pytest.raises(ValueError):
        _split(strata=(records[0], records[0]))
    with pytest.raises(ValueError):
        StratumEvidence(name="x", value="", metrics=(), support=support)


def test_split_evidence_validation() -> None:
    """SplitEvidence rejects bad hashes and duplicate names/paths."""
    with pytest.raises(ValueError):
        _split(dataset_hash="xyz")
    with pytest.raises(ValueError):
        _split(checkpoint_hash="short")
    with pytest.raises(ValueError):
        _split(dataset_manifest_hash="UP" + "0" * 62)
    two = (MetricValue(name="iou", value=0.5, status="AVAILABLE"),) * 2
    with pytest.raises(ValueError):
        _split(metrics=two)
    support = MetricSupport(unit="IMAGE", n=1)
    curve = CurveEvidence(
        name="c",
        x_name="x",
        y_name="y",
        x=(0.0,),
        y=(0.5,),
        x_unit="t",
        y_unit="t",
        support=support,
    )
    with pytest.raises(ValueError):
        _split(curves=(curve, curve))
    with pytest.raises(ValueError):
        _split(artifacts=(_artifact_ref(), _artifact_ref()))
    _split(artifacts=(_artifact_ref(path="a.csv"), _artifact_ref(path="b.csv")))
