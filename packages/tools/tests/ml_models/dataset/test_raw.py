"""Authoritative observation and annotation metadata contracts."""

from dataclasses import replace

import numpy as np
import pytest
from tools.ml_models.dataset.raw import (
    ConditionTag,
    ObservationMetadata,
    prepared_mask_state,
)


def test_unknown_metadata_is_not_fabricated() -> None:
    metadata = ObservationMetadata()
    assert metadata.observation_id is None
    assert metadata.acquired_at_utc is None
    assert metadata.conditions == ()
    assert metadata.annotation_source is None
    assert metadata.annotation_version is None
    assert metadata.source_annotation_state == "UNKNOWN"


def test_recorded_time_tags_and_annotation_provenance_round_trip() -> None:
    metadata = ObservationMetadata(
        observation_id="source-original",
        acquired_at_utc="2020-02-29T10:56:41.330Z",
        conditions=(ConditionTag(name="illumination", value="recorded-daylight"),),
        annotation_source="labels/source-original.json",
        annotation_version="annotator-v2",
        source_annotation_state="NONEMPTY",
    )
    assert replace(metadata) == metadata
    assert metadata.conditions[0].value == "recorded-daylight"


@pytest.mark.parametrize(
    "timestamp",
    [
        "2020-02-30T00:00:00Z",
        "2020-01-01",
        "2020-01-01T00:00:00",
        "2020-01-01T00:00:00+01:00",
        "2020-01-01T00-00-00.000Z",
        "2020-01-01T24:00:00Z",
    ],
)
def test_acquisition_time_requires_valid_explicit_utc(timestamp: str) -> None:
    with pytest.raises(ValueError):
        ObservationMetadata(acquired_at_utc=timestamp)


def test_empty_tags_and_duplicate_condition_names_are_rejected() -> None:
    with pytest.raises(ValueError):
        ConditionTag(name="weather", value=" ")
    with pytest.raises(ValueError):
        ObservationMetadata(observation_id="")
    with pytest.raises(ValueError):
        ObservationMetadata(
            conditions=(
                ConditionTag(name="weather", value="clear"),
                ConditionTag(name="weather", value="cloudy"),
            )
        )


def test_prepared_mask_state_is_independent_of_presence_label_and_source_annotation() -> None:
    assert prepared_mask_state(None) == "MISSING"
    empty = np.zeros((1, 2, 3), dtype=np.uint8)
    assert prepared_mask_state(empty) == "EMPTY"
    positive = empty.copy()
    positive[0, 0, 0] = 1
    assert prepared_mask_state(positive) == "NONEMPTY"
    original = ObservationMetadata(source_annotation_state="NONEMPTY")
    assert original.source_annotation_state == "NONEMPTY"
    assert prepared_mask_state(empty) == "EMPTY"
