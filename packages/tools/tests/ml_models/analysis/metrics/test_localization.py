"""Independent cardinality-first assignment and component-geometry references."""

import itertools
import math
from dataclasses import replace

import numpy as np
import pytest
from flight.libs.types import Err, Ok
from tools.ml_models.analysis.config import ScoreConfig
from tools.ml_models.analysis.metrics.definitions import metric_definition
from tools.ml_models.analysis.metrics.localization import (
    aggregate_localization,
    match_components,
    score_localization,
)
from tools.ml_models.analysis.metrics.spatial import aggregate_spatial, score_spatial


def test_cardinality_beats_a_single_higher_iou_match() -> None:
    matched = match_components(np.array([[0.9, 0.5], [0.5, 0.49]]), 0.5)
    assert isinstance(matched, Ok)
    assert matched.value == ((0, 1), (1, 0))
    tied = match_components(np.full((2, 2), 0.5), 0.5)
    assert isinstance(tied, Ok)
    assert tied.value == ((0, 0), (1, 1))


def test_match_objective_agrees_with_independent_brute_force_reference() -> None:
    rng = np.random.default_rng(17)
    for _ in range(80):
        n, m = (int(value) for value in rng.integers(0, 4, size=2))
        values = rng.integers(0, 11, size=(n, m)) / 10
        scored = match_components(values, 0.5)
        assert isinstance(scored, Ok)
        best = (0, 0.0)
        for k in range(min(n, m) + 1):
            for truths in itertools.combinations(range(n), k):
                for predictions in itertools.permutations(range(m), k):
                    pairs = tuple(zip(truths, predictions, strict=True))
                    if all(values[a, b] >= 0.5 for a, b in pairs):
                        objective = (k, math.fsum(float(values[a, b]) for a, b in pairs))
                        best = max(best, objective)
        objective = (len(scored.value), math.fsum(float(values[a, b]) for a, b in scored.value))
        assert objective[0] == best[0]
        assert objective[1] == pytest.approx(best[1])


def test_two_components_keep_unmatched_truths_and_anisotropic_centroid_error() -> None:
    truth = np.zeros((1, 5, 8), dtype=np.uint8)
    truth[0, 1:3, 1:4] = 1
    truth[0, 4, 7] = 1
    predicted = np.zeros_like(truth)
    predicted[0, 1:3, 2:5] = 1
    scored = score_localization(
        np.where(predicted, 4.0, -4.0),
        truth,
        gsd=(2.0, 3.0),
        cfg=ScoreConfig(min_blob_area_px=2, match_iou_min=0.5),
    )
    assert isinstance(scored, Ok)
    row = scored.value
    assert len(row.truth) == 2 and len(row.predicted) == 1
    assert row.truth[1].area_px == 1
    assert row.truth[0].area_m2 == 36
    assert row.unmatched_truth == (1,) and row.unmatched_prediction == ()
    assert len(row.matches) == 1
    match = row.matches[0]
    assert match.iou == 0.5
    assert match.dx_px == 1 and match.dy_px == 0
    assert match.distance_px == 1 and match.distance_m == 2
    aggregate = aggregate_localization((row,))
    assert isinstance(aggregate, Ok)
    metrics = {metric.name: metric for metric in aggregate.value.metrics}
    assert metrics["component_recall"].value == 0.5
    assert metrics["component_precision"].value == 1
    assert metrics["component_f1"].value == pytest.approx(2 / 3)
    curve = next(c for c in aggregate.value.curves if c.name == "localization_success_m")
    assert curve.x == (0, 2)
    assert curve.y == (0, 0.5)
    assert curve.support.n == 2


def test_splits_and_merges_use_retained_overlapping_components() -> None:
    truth = np.zeros((3, 8), dtype=np.uint8)
    truth[1, 1:7] = 1
    prediction = truth.copy()
    prediction[1, 3:5] = 0
    cfg = ScoreConfig(min_blob_area_px=1, match_iou_min=0.2)
    split = score_localization(np.where(prediction, 8.0, -8.0), truth, cfg=cfg)
    merged = score_localization(np.where(truth, 8.0, -8.0), prediction, cfg=cfg)
    assert isinstance(split, Ok) and isinstance(merged, Ok)
    assert split.value.split_truth_components == 1 and split.value.merge_predicted_components == 0
    assert merged.value.split_truth_components == 0 and merged.value.merge_predicted_components == 1
    assert len(split.value.matches) == 1 and len(split.value.unmatched_prediction) == 1
    assert len(merged.value.matches) == 1 and len(merged.value.unmatched_truth) == 1


def test_empty_components_and_missing_gsd_remain_unavailable() -> None:
    empty = np.zeros((2, 3), dtype=np.uint8)
    row = score_localization(np.full((2, 3), -1.0), empty)
    assert isinstance(row, Ok)
    aggregate = aggregate_localization((row.value,))
    assert isinstance(aggregate, Ok)
    assert all(
        metric.value is None
        for metric in aggregate.value.metrics
        if metric.name
        in (
            "component_precision",
            "component_recall",
            "component_f1",
            "matched_centroid_error_px_mean",
        )
    )
    truth = empty.copy()
    truth[0, 0] = 1
    missed = score_localization(np.full((2, 3), -1.0), truth)
    assert isinstance(missed, Ok)
    aggregate = aggregate_localization((missed.value,))
    assert isinstance(aggregate, Ok)
    metrics = {metric.name: metric for metric in aggregate.value.metrics}
    assert metrics["component_recall"].value == 0
    assert metrics["component_f1"].value == 0
    assert metrics["matched_centroid_error_px_mean"].value is None
    assert metrics["matched_centroid_error_m_mean"].value is None
    assert isinstance(score_localization(np.ones((2, 3)), empty, gsd=(0.0, 1.0)), Err)
    assert isinstance(score_localization(np.full((2, 3), np.nan), empty), Err)
    assert isinstance(match_components(np.ones((2, 2)), -1.0), Err)


def test_mixed_geometry_success_denominator_does_not_drop_known_ground_misses() -> None:
    truth = np.zeros((2, 3), dtype=np.uint8)
    truth[0, 0] = 1
    cfg = ScoreConfig(min_blob_area_px=1)
    perfect = score_localization(np.where(truth, 8.0, -8.0), truth, gsd=(2.0, 3.0), cfg=cfg)
    missed = score_localization(np.full((2, 3), -8.0), truth, gsd=(2.0, 3.0), cfg=cfg)
    unknown = score_localization(np.where(truth, 8.0, -8.0), truth, cfg=cfg)
    assert isinstance(perfect, Ok) and isinstance(missed, Ok) and isinstance(unknown, Ok)
    measured = aggregate_localization((perfect.value, missed.value, unknown.value))
    assert isinstance(measured, Ok)
    curves = {c.name: c for c in measured.value.curves}
    assert curves["localization_success_m"].support.n == 2
    assert curves["localization_success_m"].x == (0,)
    assert curves["localization_success_m"].y == (0.5,)
    assert {
        count.name: count.value for count in curves["localization_success_m"].support.counts
    } == {
        "total_truth_components": 3,
        "truth_components_missing_geometry": 1,
    }
    assert curves["localization_success_px"].support.n == 3
    assert curves["localization_success_px"].y == (2 / 3,)


def test_spatial_extent_and_blob_gates_remain_separate_and_definitions_complete() -> None:
    truth = np.zeros((2, 3), dtype=np.uint8)
    truth[0, 0] = 1
    row = score_spatial(np.where(truth, 1.0, -1.0), truth, gsd=(2.0, 3.0))
    assert isinstance(row, Ok)
    assert len(row.value.localization.truth) == 1
    assert len(row.value.localization.predicted) == 0
    assert row.value.boundary.truth_hits == row.value.boundary.predicted_hits == 1
    measured = aggregate_spatial((row.value,))
    assert isinstance(measured, Ok)
    for metric in measured.value.metrics:
        defined = metric_definition(metric.name)
        assert isinstance(defined, Ok), metric.name
        assert defined.value.unit == metric.unit
        assert defined.value.aggregation == metric.aggregation


def test_corrupt_component_assignments_fail_before_summary_rates() -> None:
    mask = np.ones((2, 3), dtype=np.uint8)
    measured = score_localization(
        np.full((2, 3), 4.0),
        mask,
        cfg=ScoreConfig(min_blob_area_px=1),
    )
    assert isinstance(measured, Ok)
    row = measured.value
    assert isinstance(aggregate_localization((replace(row, matches=row.matches * 2),)), Err)
    assert isinstance(
        aggregate_localization(
            (
                replace(
                    row,
                    matches=(replace(row.matches[0], distance_px=float("nan")),),
                ),
            )
        ),
        Err,
    )


def test_mixed_raw_mask_thresholds_cannot_merge_boundary_evidence() -> None:
    truth = np.ones((2, 3), dtype=np.uint8)
    first = score_spatial(np.ones((2, 3)), truth, cfg=ScoreConfig(mask_probability_threshold=0.5))
    second = score_spatial(np.ones((2, 3)), truth, cfg=ScoreConfig(mask_probability_threshold=0.9))
    assert isinstance(first, Ok) and isinstance(second, Ok)
    assert isinstance(aggregate_spatial((first.value, second.value)), Err)
