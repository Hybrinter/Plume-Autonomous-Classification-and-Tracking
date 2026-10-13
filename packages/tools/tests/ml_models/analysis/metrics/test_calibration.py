"""Probability calibration reference values and exact bin-edge conventions."""

import numpy as np
import pytest
from flight.libs.types import Err, Ok
from tools.ml_models.analysis.metrics.calibration import score_calibration


def test_calibration_oracle() -> None:
    """Four explicit probabilities have Brier 0.02 and ECE 0.1."""
    result = score_calibration([0.0, 0.2, 0.8, 1.0], [0.0, 0.0, 1.0, 1.0], n_bins=2)
    assert isinstance(result, Ok)
    metrics = {metric.name: metric.value for metric in result.value.metrics}
    assert metrics["brier_score"] == pytest.approx(0.02)
    assert metrics["expected_calibration_error"] == pytest.approx(0.1)
    assert tuple(bin.n for bin in result.value.bins) == (2, 2)
    assert tuple(bin.mean_probability for bin in result.value.bins) == pytest.approx((0.1, 0.9))
    assert tuple(bin.positive_fraction for bin in result.value.bins) == (0.0, 1.0)


@pytest.mark.parametrize("n_bins", [2, 10, 100])
def test_all_calibration_edges_are_left_closed(n_bins: int) -> None:
    """Every exact i/n lower edge enters its own bin; one enters the last."""
    probabilities = np.arange(n_bins + 1, dtype=np.float64) / n_bins
    result = score_calibration(probabilities, np.zeros(n_bins + 1), n_bins=n_bins)
    assert isinstance(result, Ok)
    assert tuple(bin.n for bin in result.value.bins) == (1,) * (n_bins - 1) + (2,)
    assert sum(bin.n for bin in result.value.bins) == n_bins + 1


def test_empty_bins_and_permutation() -> None:
    """Empty bins retain null measurements and occupied-bin scores are invariant."""
    first = score_calibration([0.5, 0.5], [1.0, 0.0], n_bins=10)
    second = score_calibration([0.5, 0.5], [0.0, 1.0], n_bins=10)
    assert isinstance(first, Ok) and isinstance(second, Ok)
    assert first.value == second.value
    assert first.value.bins[5].n == 2
    assert first.value.bins[0].mean_probability is None
    assert first.value.bins[0].positive_fraction is None
    assert {metric.name: metric.value for metric in first.value.metrics} == {
        "brier_score": 0.25,
        "expected_calibration_error": 0.0,
    }


def test_invalid_calibration_inputs() -> None:
    """Out-of-domain probability/label arrays fail rather than getting clipped."""
    for probabilities, labels in (
        ([], []),
        ([-0.1], [0.0]),
        ([1.1], [1.0]),
        ([float("nan")], [0.0]),
        ([0.5], [0.25]),
        ([0.5], [1.0, 0.0]),
    ):
        assert isinstance(score_calibration(probabilities, labels), Err)
    assert isinstance(score_calibration([0.5], [1.0], n_bins=True), Err)
    assert isinstance(score_calibration([0.5], [1.0], n_bins=1), Err)
