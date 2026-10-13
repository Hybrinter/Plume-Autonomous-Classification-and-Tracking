"""Independent boundary definitions, distance and empty-support references."""

from dataclasses import replace

import numpy as np
import pytest
from flight.libs.types import Err, Ok
from tools.ml_models.analysis.config import ScoreConfig
from tools.ml_models.analysis.metrics.boundary import aggregate_boundary, score_boundary


def test_singleton_translation_has_known_pixel_and_anisotropic_ground_distance() -> None:
    truth = np.zeros((4, 5), dtype=np.uint8)
    predicted = truth.copy()
    truth[1, 1] = 1
    predicted[1, 2] = 1
    row = score_boundary(predicted, truth, gsd=(2.0, 3.0))
    assert isinstance(row, Ok)
    result = row.value
    assert result.truth_boundary_pixels == result.predicted_boundary_pixels == 1
    assert result.truth_hits == result.predicted_hits == 1
    assert result.asd_px == result.hd95_px == 1
    assert result.asd_m == result.hd95_m == 2
    aggregate = aggregate_boundary((result,))
    assert isinstance(aggregate, Ok)
    values = {m.name: m.value for m in aggregate.value.metrics}
    assert values["boundary_precision"] == values["boundary_recall"] == values["boundary_f1"] == 1
    smaller = score_boundary(
        predicted,
        truth,
        gsd=(2.0, 3.0),
        cfg=ScoreConfig(boundary_tolerance_m=1.0),
    )
    assert isinstance(smaller, Ok)
    assert smaller.value.truth_hits == smaller.value.predicted_hits == 0
    assert smaller.value.tolerance_unit == "m"


def test_identical_full_mask_boundary_includes_image_border_and_excludes_interior() -> None:
    mask = np.ones((3, 4), dtype=np.uint8)
    row = score_boundary(mask, mask, gsd=(1.0, 2.0))
    assert isinstance(row, Ok)
    assert row.value.truth_boundary_pixels == 10
    assert row.value.truth_hits == row.value.predicted_hits == 10
    assert row.value.asd_px == row.value.hd95_px == row.value.asd_m == row.value.hd95_m == 0


def test_pooled_directed_distances_match_independent_pairwise_reference() -> None:
    truth = np.zeros((5, 7), dtype=np.uint8)
    predicted = truth.copy()
    truth[1, 1:4] = 1
    predicted[3, 1] = 1
    predicted[1, 5] = 1
    points_a, points_b = np.argwhere(truth), np.argwhere(predicted)
    distances = np.sqrt(((points_a[:, None] - points_b[None, :]) ** 2).sum(axis=2))
    pooled = np.concatenate((distances.min(axis=1), distances.min(axis=0)))
    row = score_boundary(predicted, truth)
    assert isinstance(row, Ok)
    assert row.value.asd_px == pytest.approx(pooled.mean())
    assert row.value.hd95_px == pytest.approx(np.quantile(pooled, 0.95, method="linear"))
    assert row.value.asd_m is None


def test_empty_boundaries_report_misses_without_fabricated_distances() -> None:
    empty = np.zeros((2, 3), dtype=np.uint8)
    singleton = empty.copy()
    singleton[0, 0] = 1
    both = score_boundary(empty, empty)
    missed = score_boundary(empty, singleton)
    assert isinstance(both, Ok) and isinstance(missed, Ok)
    assert both.value.distance_reason and missed.value.distance_reason
    assert both.value.asd_px is None and missed.value.hd95_px is None
    aggregate = aggregate_boundary((missed.value,))
    assert isinstance(aggregate, Ok)
    metrics = {m.name: m for m in aggregate.value.metrics}
    assert metrics["boundary_precision"].value is None
    assert metrics["boundary_recall"].value == 0
    assert metrics["boundary_f1"].value == 0
    assert metrics["boundary_missed_images"].value == 1
    assert metrics["boundary_asd_px_mean"].value is None
    assert isinstance(score_boundary(empty, empty, cfg=ScoreConfig(boundary_tolerance_m=1.0)), Err)
    assert isinstance(score_boundary(empty.astype(np.float64) + 0.1, empty), Err)


def test_corrupt_frozen_boundary_rows_fail_instead_of_producing_plausible_rates() -> None:
    mask = np.ones((2, 3), dtype=np.uint8)
    measured = score_boundary(mask, mask)
    assert isinstance(measured, Ok)
    assert isinstance(aggregate_boundary((replace(measured.value, truth_hits=7),)), Err)
    assert isinstance(aggregate_boundary((replace(measured.value, asd_px=float("nan")),)), Err)
