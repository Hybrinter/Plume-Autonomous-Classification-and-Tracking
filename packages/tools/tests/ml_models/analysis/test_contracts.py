"""Tests for the frozen evidence-contract records."""

import dataclasses

import pytest
from tools.ml_models.analysis.contracts import MetricValue, SampleKey, SplitEvidence


def test_sample_key_fields() -> None:
    """SampleKey carries the full row-alignment identity."""
    key = SampleKey(
        dataset_hash="a" * 64,
        task="classifier",
        split="val",
        spatial_shard=(193, 258),
        row_index=3,
        tile_id="tile-1",
        element="id",
    )
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


def test_metric_value_unavailable_reason() -> None:
    """A missing measurement is explicit, never a fabricated number."""
    unavailable = MetricValue(name="roc_auc", value=None, status="UNAVAILABLE", reason="one class")
    assert unavailable.value is None
    assert unavailable.reason == "one class"
    available = MetricValue(name="accuracy", value=0.9, status="AVAILABLE")
    assert available.reason is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        available.value = 0.1  # type: ignore[misc]


def test_split_evidence_defaults_to_empty_metrics() -> None:
    """SplitEvidence carries task, split, and dataset identity."""
    evidence = SplitEvidence(task="segmentor", split="test", dataset_hash="b" * 64)
    assert evidence.metrics == ()
    named = SplitEvidence(
        task="segmentor",
        split="test",
        dataset_hash="b" * 64,
        metrics=(MetricValue(name="iou", value=0.5, status="AVAILABLE"),),
    )
    assert named.metrics[0].name == "iou"
