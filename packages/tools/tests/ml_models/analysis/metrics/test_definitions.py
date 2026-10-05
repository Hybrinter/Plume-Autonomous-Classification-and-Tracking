"""Non-numeric checks for metric definitions and evidence serialization."""

import dataclasses
import json

import pytest
from flight.libs.types import Err, Ok
from tools.ml_models.analysis.metrics.calibration import CalibrationEvidence, ReliabilityBin
from tools.ml_models.analysis.metrics.classifier import (
    ClassifierEvidence,
    ConfusionCounts,
    score_classifier,
    score_classifier_baseline,
)
from tools.ml_models.analysis.metrics.definitions import (
    CALIBRATION_DEFINITIONS,
    CLASSIFIER_DEFINITIONS,
    SEGMENTATION_DEFINITIONS,
    MetricDefinition,
    metric_definition,
)
from tools.ml_models.analysis.metrics.segmentation import (
    aggregate_segmentation,
    score_segmentation_image,
)

_ALL_DEFINITIONS = CLASSIFIER_DEFINITIONS + CALIBRATION_DEFINITIONS + SEGMENTATION_DEFINITIONS


def test_metric_definition_lookup() -> None:
    """Every declared name resolves; unknown names are explicit errors."""
    for definition in _ALL_DEFINITIONS:
        found = metric_definition(definition.name)
        assert isinstance(found, Ok)
        assert found.value is definition
    assert isinstance(metric_definition("not_a_metric"), Err)
    assert isinstance(metric_definition(""), Err)


def test_definitions_are_complete_metadata() -> None:
    """Each definition carries direction, formula, and limitation text."""
    for definition in _ALL_DEFINITIONS:
        assert definition.direction in ("MINIMIZE", "MAXIMIZE", "DESCRIPTIVE")
        assert definition.name.isidentifier() or "_" in definition.name
        assert definition.formula.strip()
        assert definition.population.strip()
        assert definition.aggregation.strip()
        assert definition.undefined_policy.strip()
        assert definition.limitations.strip()
        assert definition.unit.strip()
    names = [definition.name for definition in _ALL_DEFINITIONS]
    assert len(set(names)) == len(names)


def test_produced_metrics_have_definitions() -> None:
    """Every metric a scorer emits has an explicit definition record."""
    scored = score_classifier([0.5, -1.0, 2.0], [1, 0, 1])
    assert isinstance(scored, Ok)
    for metric in scored.value.metrics:
        assert isinstance(metric_definition(metric.name), Ok), metric.name
    base = score_classifier_baseline(0.5, [1, 0])
    assert isinstance(base, Ok)
    for metric in base.value.metrics:
        assert isinstance(metric_definition(metric.name), Ok), metric.name


def test_segmentation_metrics_have_definitions() -> None:
    """Every aggregated segmentation metric name resolves to a definition."""
    rows = []
    for logits, mask, label, empty in (
        ([[2.0, -2.0]], [[1, 0]], 1.0, False),
        ([[-2.0, -2.0]], [[0, 0]], 0.0, True),
    ):
        scored = score_segmentation_image(logits, mask, label=label, verified_empty=empty)
        assert isinstance(scored, Ok)
        rows.append(scored.value)
    aggregate = aggregate_segmentation(tuple(rows))
    assert isinstance(aggregate, Ok)
    assert aggregate.value.metrics
    for metric in aggregate.value.metrics:
        assert isinstance(metric_definition(metric.name), Ok), metric.name


def test_evidence_records_are_frozen_and_serializable() -> None:
    """Evidence dataclasses are immutable and convert to plain data."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        ConfusionCounts(tp=1, fp=0, tn=0, fn=0).tp = 2  # type: ignore[misc]
    empty_bin = ReliabilityBin(
        lower=0.0, upper=0.5, n=0, mean_probability=None, positive_fraction=None
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        empty_bin.n = 1  # type: ignore[misc]
    scored = score_classifier([0.5, -1.0], [1, 0])
    assert isinstance(scored, Ok)
    evidence = scored.value
    assert isinstance(evidence, ClassifierEvidence)
    assert isinstance(evidence.calibration, CalibrationEvidence)
    plain = dataclasses.asdict(evidence.counts)
    assert set(plain) == {"tp", "fp", "tn", "fn"}
    assert sum(plain.values()) == evidence.support.n
    with pytest.raises(dataclasses.FrozenInstanceError):
        evidence.counts = ConfusionCounts(tp=0, fp=0, tn=0, fn=0)  # type: ignore[misc]
    json.dumps(dataclasses.asdict(evidence.counts))
    json.dumps(dataclasses.asdict(evidence.calibration.bins[0]))
    found = metric_definition("roc_auc")
    assert isinstance(found, Ok)
    assert isinstance(found.value, MetricDefinition)
