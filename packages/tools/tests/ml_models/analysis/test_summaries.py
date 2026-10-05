"""Tests for the tagged, versioned summary records and JSON codecs."""

import json

import pytest
from flight.libs.types import Err, Ok
from tools.ml_models.analysis.artifacts import decode_summary, encode_summary
from tools.ml_models.analysis.contracts import (
    ArtifactRef,
    AvailabilityRecord,
    CheckpointIdentity,
    CodeIdentity,
    DatasetIdentity,
    MetricValue,
    SplitEvidence,
)
from tools.ml_models.analysis.summaries import DatasetSummary, ModelTrainingSummary

_HEX_A = "a" * 64
_HEX_B = "b" * 64
_HEX_C = "c" * 64
_HEX_D = "d" * 64
_HEX_E = "e" * 64


def _dataset(content: str = _HEX_B, manifest: str = _HEX_C) -> DatasetIdentity:
    return DatasetIdentity(
        content_hash=content,
        manifest_hash=manifest,
        schema_version=2,
        source="fixture",
        band_names=("red", "nir"),
        gsd_reference_m=0.05,
    )


def _code() -> CodeIdentity:
    return CodeIdentity(revision=None, dirty=None, reason="git unavailable")


def _checkpoint(training_hash: str = _HEX_B) -> CheckpointIdentity:
    return CheckpointIdentity(
        sha256=_HEX_D,
        kind="segmentor",
        arch="unet",
        training_dataset_hash=training_hash,
        epoch=3,
    )


def _dataset_summary(**overrides: object) -> DatasetSummary:
    values: dict[str, object] = {
        "measurement_id": _HEX_A,
        "dataset": _dataset(),
        "code": _code(),
        "config_digest": _HEX_E,
    }
    values.update(overrides)
    return DatasetSummary(**values)  # type: ignore[arg-type]


def _model_summary(**overrides: object) -> ModelTrainingSummary:
    values: dict[str, object] = {
        "measurement_id": _HEX_A,
        "training_dataset": _dataset(),
        "evaluation_dataset": _dataset(),
        "checkpoint": _checkpoint(),
        "code": _code(),
        "config_digest": _HEX_E,
    }
    values.update(overrides)
    return ModelTrainingSummary(**values)  # type: ignore[arg-type]


def _split(dataset: str = _HEX_B) -> SplitEvidence:
    return SplitEvidence(
        task="segmentor",
        split="test",
        dataset_hash=dataset,
        metrics=(MetricValue(name="iou", value=0.5, status="AVAILABLE"),),
    )


def _ref(path: str = "tables/metrics.csv") -> ArtifactRef:
    return ArtifactRef(
        path=path,
        sha256=_HEX_C,
        size_bytes=12,
        kind="TABLE",
        format="csv",
        rows=3,
    )


def test_dataset_summary_minimal() -> None:
    """A minimal dataset summary constructs with tagged defaults."""
    summary = _dataset_summary()
    assert summary.status == "COMPLETE"
    assert summary.schema_version == 1
    assert summary.summary_kind == "DATASET_ANALYSIS"
    assert summary.splits == ()


def test_model_summary_minimal() -> None:
    """A minimal model/training summary constructs with its tag."""
    summary = _model_summary()
    assert summary.summary_kind == "MODEL_TRAINING_ANALYSIS"
    assert summary.status == "COMPLETE"
    assert summary.history is None


def test_summary_digest_validation() -> None:
    """Measurement and config digests are strict SHA-256 values."""
    with pytest.raises(ValueError):
        _dataset_summary(measurement_id="not-a-hash")
    with pytest.raises(ValueError):
        _dataset_summary(config_digest="A" * 64)
    with pytest.raises(ValueError):
        _model_summary(measurement_id="x")


def test_summary_status_and_required_outputs() -> None:
    """COMPLETE summaries cannot carry required unavailable/skipped outputs."""
    required_missing = AvailabilityRecord(
        name="figures",
        status="UNAVAILABLE",
        reason="plot failed",
        required=True,
    )
    with pytest.raises(ValueError):
        _dataset_summary(outputs=(required_missing,))
    partial = _dataset_summary(outputs=(required_missing,), status="PARTIAL")
    assert partial.status == "PARTIAL"
    failed = _dataset_summary(outputs=(required_missing,), status="FAILED")
    assert failed.status == "FAILED"
    optional_missing = AvailabilityRecord(
        name="predictions",
        status="SKIPPED",
        reason="compact capture",
        required=False,
    )
    complete = _dataset_summary(outputs=(optional_missing,))
    assert complete.status == "COMPLETE"
    with pytest.raises(ValueError):
        _dataset_summary(status="DONE")
    with pytest.raises(ValueError):
        _dataset_summary(schema_version=2)
    with pytest.raises(ValueError):
        _dataset_summary(summary_kind="MODEL_TRAINING_ANALYSIS")


def test_summary_split_identity_matching() -> None:
    """Split records must agree with the dataset they describe."""
    summary = _dataset_summary(splits=(_split(),))
    assert summary.splits[0].dataset_hash == _HEX_B
    with pytest.raises(ValueError):
        _dataset_summary(splits=(_split(dataset=_HEX_D),))
    split_with_manifest = SplitEvidence(
        task="segmentor",
        split="val",
        dataset_hash=_HEX_B,
        dataset_manifest_hash=_HEX_C,
    )
    assert _dataset_summary(splits=(split_with_manifest,)).splits[0].split == "val"
    bad_manifest = SplitEvidence(
        task="segmentor",
        split="val",
        dataset_hash=_HEX_B,
        dataset_manifest_hash=_HEX_D,
    )
    with pytest.raises(ValueError):
        _dataset_summary(splits=(bad_manifest,))
    checkpoint_split = SplitEvidence(
        task="segmentor",
        split="val",
        dataset_hash=_HEX_B,
        checkpoint_hash=_HEX_D,
    )
    with pytest.raises(ValueError):
        _dataset_summary(splits=(checkpoint_split,))


def test_model_summary_identity_matching() -> None:
    """Model summaries bind checkpoint and evaluation-dataset identities."""
    good_split = SplitEvidence(
        task="segmentor",
        split="val",
        dataset_hash=_HEX_B,
        dataset_manifest_hash=_HEX_C,
        checkpoint_hash=_HEX_D,
    )
    summary = _model_summary(splits=(good_split,))
    assert summary.splits[0].checkpoint_hash == _HEX_D
    with pytest.raises(ValueError):
        _model_summary(checkpoint=_checkpoint(training_hash=_HEX_D))
    with pytest.raises(ValueError):
        _model_summary(splits=(_split(dataset=_HEX_D),))
    mixed_task = SplitEvidence(
        task="classifier",
        split="val",
        dataset_hash=_HEX_B,
        dataset_manifest_hash=_HEX_C,
        checkpoint_hash=_HEX_D,
    )
    with pytest.raises(ValueError):
        _model_summary(splits=(mixed_task,))
    wrong_checkpoint = SplitEvidence(
        task="segmentor",
        split="val",
        dataset_hash=_HEX_B,
        checkpoint_hash=_HEX_E,
    )
    with pytest.raises(ValueError):
        _model_summary(splits=(wrong_checkpoint,))
    wrong_manifest = SplitEvidence(
        task="segmentor",
        split="val",
        dataset_hash=_HEX_B,
        dataset_manifest_hash=_HEX_D,
    )
    with pytest.raises(ValueError):
        _model_summary(splits=(wrong_manifest,))


def test_summary_uniqueness_rules() -> None:
    """Metric names, output names, split identities, and paths are unique."""
    metrics = (MetricValue(name="iou", value=0.5, status="AVAILABLE"),) * 2
    with pytest.raises(ValueError):
        _dataset_summary(metrics=metrics)
    outputs = (
        AvailabilityRecord(name="x", status="AVAILABLE"),
        AvailabilityRecord(name="x", status="AVAILABLE"),
    )
    with pytest.raises(ValueError):
        _dataset_summary(outputs=outputs)
    with pytest.raises(ValueError):
        _dataset_summary(artifacts=(_ref(), _ref()))
    with pytest.raises(ValueError):
        _dataset_summary(splits=(_split(), _split()))
    two_splits = (_split(), SplitEvidence(task="segmentor", split="val", dataset_hash=_HEX_B))
    assert len(_dataset_summary(splits=two_splits).splits) == 2


def test_summary_codec_roundtrips() -> None:
    """Both summary kinds survive the canonical JSON codec."""
    dataset_summary = _dataset_summary(
        metrics=(MetricValue(name="coverage", value=0.9, status="AVAILABLE"),),
        splits=(_split(),),
        artifacts=(_ref(),),
        outputs=(AvailabilityRecord(name="predictions", status="SKIPPED", reason="compact"),),
        warnings=("narrow gsd window",),
    )
    encoded = encode_summary(dataset_summary)
    assert isinstance(encoded, Ok)
    payload = json.loads(encoded.value.decode("utf-8"))
    assert payload["summary_kind"] == "DATASET_ANALYSIS"
    assert payload["schema_version"] == 1
    decoded = decode_summary(encoded.value)
    assert isinstance(decoded, Ok)
    assert decoded.value == dataset_summary

    model_summary = _model_summary(
        splits=(
            SplitEvidence(
                task="segmentor",
                split="test",
                dataset_hash=_HEX_B,
                checkpoint_hash=_HEX_D,
                metrics=(MetricValue(name="iou", value=0.4, status="AVAILABLE"),),
            ),
        ),
        history=ArtifactRef(
            path="history.jsonl",
            sha256=_HEX_E,
            size_bytes=3,
            kind="REFERENCE",
            format="jsonl",
        ),
    )
    encoded_model = encode_summary(model_summary)
    assert isinstance(encoded_model, Ok)
    decoded_model = decode_summary(encoded_model.value)
    assert isinstance(decoded_model, Ok)
    assert decoded_model.value == model_summary


def test_summary_codec_rejects_bad_payloads() -> None:
    """Decode rejects unknown kinds, future versions, extras, and NaN."""
    encoded = encode_summary(_dataset_summary())
    assert isinstance(encoded, Ok)
    payload = json.loads(encoded.value.decode("utf-8"))

    unknown = dict(payload, summary_kind="FUTURE_KIND")
    result = decode_summary(json.dumps(unknown).encode())
    assert isinstance(result, Err)

    future = dict(payload, schema_version=99)
    result = decode_summary(json.dumps(future).encode())
    assert isinstance(result, Err)

    extra = dict(payload, surprise=1)
    result = decode_summary(json.dumps(extra).encode())
    assert isinstance(result, Err)

    wrong_kind_fields = dict(payload, summary_kind="MODEL_TRAINING_ANALYSIS")
    result = decode_summary(json.dumps(wrong_kind_fields).encode())
    assert isinstance(result, Err)

    nan_metric = dict(payload)
    nan_metric["metrics"] = [
        {"name": "m", "value": float("nan"), "status": "AVAILABLE"},
    ]
    with pytest.raises(ValueError):
        json.dumps(nan_metric, allow_nan=False)
    result = decode_summary(
        b'{"summary_kind":"DATASET_ANALYSIS","metrics":[{"name":"m","value":NaN}]}'
    )
    assert isinstance(result, Err)

    assert isinstance(decode_summary(b"not json"), Err)
    assert isinstance(decode_summary(b"[1,2]"), Err)
    assert isinstance(decode_summary(b"\xff\xfe"), Err)


def test_summary_codec_counts_stay_exact() -> None:
    """Float and bool counts are rejected at the JSON boundary."""
    encoded = encode_summary(_dataset_summary())
    assert isinstance(encoded, Ok)
    payload = json.loads(encoded.value.decode("utf-8"))
    payload["splits"] = [
        {
            "task": "segmentor",
            "split": "test",
            "dataset_hash": _HEX_B,
            "metrics": [],
            "curves": [],
            "artifacts": [],
            "support": {"unit": "IMAGE", "n": True, "counts": []},
            "warnings": [],
        }
    ]
    result = decode_summary(json.dumps(payload).encode())
    assert isinstance(result, Err)


def test_encode_summary_is_canonical() -> None:
    """Encoding is deterministic with sorted keys."""
    summary = _dataset_summary(warnings=("b", "a"))
    first = encode_summary(summary)
    second = encode_summary(summary)
    assert isinstance(first, Ok) and isinstance(second, Ok)
    assert first.value == second.value
    text = first.value.decode("utf-8")
    keys = list(json.loads(text).keys())
    assert keys == sorted(keys)
