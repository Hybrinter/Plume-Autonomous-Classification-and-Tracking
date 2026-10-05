"""Independent binary-score oracles for the shared classifier core."""

import math

import numpy as np
import pytest
from flight.libs.types import Err, Ok
from tools.ml_models.analysis.config import ScoreConfig
from tools.ml_models.analysis.metrics.classifier import (
    ClassifierEvidence,
    score_classifier,
    score_classifier_baseline,
)


def _scores(evidence: ClassifierEvidence) -> dict[str, float | None]:
    """Read the named values without deriving expected scores from the core."""
    return {metric.name: metric.value for metric in evidence.metrics}


@pytest.mark.parametrize("labels", [(1.0, 0.0), (0.0, 1.0)])
def test_equal_scores_have_order_independent_ranking(labels: tuple[float, float]) -> None:
    """A complete positive/negative tie has ROC 0.5 and AP 0.5."""
    result = score_classifier([0.0, 0.0], labels)
    assert isinstance(result, Ok)
    scores = _scores(result.value)
    assert scores["roc_auc"] == pytest.approx(0.5)
    assert scores["average_precision"] == pytest.approx(0.5)
    assert scores["accuracy"] == pytest.approx(0.5)
    assert scores["precision"] == pytest.approx(0.5)
    assert scores["recall"] == pytest.approx(1.0)
    assert scores["f1"] == pytest.approx(2 / 3)
    assert scores["binary_cross_entropy"] == pytest.approx(math.log(2))
    assert scores["brier_score"] == pytest.approx(0.25)


@pytest.mark.parametrize("order", [(0, 1, 2, 3, 4), (4, 3, 2, 1, 0), (1, 0, 4, 3, 2)])
def test_grouped_tie_oracle(order: tuple[int, ...]) -> None:
    """Three complete score groups have hand-counted ROC 1/2 and AP 53/90."""
    logits = np.array([3.0, 3.0, 2.0, 1.0, 1.0])
    labels = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
    result = score_classifier(logits[list(order)], labels[list(order)])
    assert isinstance(result, Ok)
    scores = _scores(result.value)
    assert scores["roc_auc"] == pytest.approx(1 / 2)
    assert scores["average_precision"] == pytest.approx(53 / 90)
    curves = {curve.name: curve for curve in result.value.curves}
    assert curves["roc"].x == pytest.approx((0.0, 0.5, 0.5, 1.0))
    assert curves["roc"].y == pytest.approx((0.0, 1 / 3, 2 / 3, 1.0))
    assert curves["roc"].thresholds == (None, 3.0, 2.0, 1.0)
    assert curves["precision_recall"].x == pytest.approx((0.0, 1 / 3, 2 / 3, 1.0))
    assert curves["cumulative_gain"].x[-1] == 1.0
    assert curves["cumulative_gain"].y[-1] == 1.0
    assert curves["lift"].y[0] is None


def test_perfect_and_reversed_rankings() -> None:
    """Rank reversal produces ROC zero and AP 5/12, not one minus AP."""
    perfect = score_classifier([2.0, 1.0, -1.0, -2.0], [1.0, 1.0, 0.0, 0.0])
    reverse = score_classifier([2.0, 1.0, -1.0, -2.0], [0.0, 0.0, 1.0, 1.0])
    assert isinstance(perfect, Ok) and isinstance(reverse, Ok)
    assert _scores(perfect.value)["roc_auc"] == 1.0
    assert _scores(perfect.value)["average_precision"] == 1.0
    assert _scores(reverse.value)["roc_auc"] == 0.0
    assert _scores(reverse.value)["average_precision"] == pytest.approx(5 / 12)


def test_confusion_metrics_have_distinct_denominators() -> None:
    """TP=1, FN=1, TN=2, FP=0 identifies each operating-point denominator."""
    result = score_classifier([2.0, -2.0, -2.0, -2.0], [1.0, 1.0, 0.0, 0.0])
    assert isinstance(result, Ok)
    scores = _scores(result.value)
    assert result.value.counts.tp == 1
    assert result.value.counts.fn == 1
    assert result.value.counts.tn == 2
    assert result.value.counts.fp == 0
    expected = {
        "accuracy": 0.75,
        "precision": 1.0,
        "recall": 0.5,
        "specificity": 1.0,
        "negative_predictive_value": 2 / 3,
        "false_positive_rate": 0.0,
        "false_negative_rate": 0.5,
        "f1": 2 / 3,
        "balanced_accuracy": 0.75,
        "matthews_correlation": 1 / math.sqrt(3),
        "predicted_positive_fraction": 0.25,
    }
    for name, value in expected.items():
        assert scores[name] == pytest.approx(value)
    support = {count.name: count.value for count in result.value.support.counts}
    assert support == {"n_positive": 2, "n_negative": 2}


def test_undefined_values_are_not_zero_scores() -> None:
    """No positives/no positive predictions is distinct from failed detection."""
    negative = score_classifier([-2.0, -2.0], [0.0, 0.0])
    missed = score_classifier([-2.0, -2.0], [1.0, 0.0])
    positive = score_classifier([2.0, 2.0], [1.0, 1.0])
    assert isinstance(negative, Ok) and isinstance(missed, Ok) and isinstance(positive, Ok)
    scores = _scores(negative.value)
    for name in ("precision", "recall", "f1", "roc_auc", "average_precision", "balanced_accuracy"):
        assert scores[name] is None
        metric = next(metric for metric in negative.value.metrics if metric.name == name)
        assert metric.reason
    assert _scores(missed.value)["f1"] == 0.0
    assert _scores(positive.value)["average_precision"] == 1.0
    assert _scores(positive.value)["roc_auc"] is None


def test_extreme_logits_preserve_ranking_and_endpoint_decisions() -> None:
    """Saturated sigmoid values do not collapse logit ranking or threshold one."""
    result = score_classifier([1000.0, 999.0, -999.0, -1000.0], [1.0, 0.0, 1.0, 0.0])
    assert isinstance(result, Ok)
    assert _scores(result.value)["average_precision"] == pytest.approx(5 / 6)
    assert _scores(result.value)["binary_cross_entropy"] == pytest.approx(499.5)
    high = score_classifier([1000.0], [1.0], ScoreConfig(classifier_probability_threshold=1.0))
    low = score_classifier([-1000.0], [0.0], ScoreConfig(classifier_probability_threshold=0.0))
    assert isinstance(high, Ok) and isinstance(low, Ok)
    assert high.value.counts.fn == 1
    assert low.value.counts.fp == 1
    curve = next(curve for curve in result.value.curves if curve.name == "threshold_precision")
    assert curve.x[0] == 0.0 and curve.x[-1] == 1.0
    assert curve.y[-1] is None


def test_f_beta_and_extreme_beta_are_stable() -> None:
    """F-beta weighting and its numerical limits remain finite."""
    two = score_classifier([0.0, 0.0], [1.0, 0.0], ScoreConfig(f_beta=2.0))
    large = score_classifier([0.0, 0.0], [1.0, 0.0], ScoreConfig(f_beta=1e200))
    tiny = score_classifier([0.0, 0.0], [1.0, 0.0], ScoreConfig(f_beta=1e-200))
    assert isinstance(two, Ok) and isinstance(large, Ok) and isinstance(tiny, Ok)
    assert _scores(two.value)["f_beta"] == pytest.approx(5 / 6)
    assert _scores(large.value)["f_beta"] == pytest.approx(1.0)
    assert _scores(tiny.value)["f_beta"] == pytest.approx(0.5)


@pytest.mark.parametrize("beta", [1e-200, 1.0, 1e200])
@pytest.mark.parametrize("logits,labels", [([2.0], [0.0]), ([-2.0], [1.0])])
def test_f_beta_error_only_is_zero_for_every_positive_beta(
    beta: float,
    logits: list[float],
    labels: list[float],
) -> None:
    """Zero TP with FP or FN has a nonzero mathematical denominator."""
    result = score_classifier(logits, labels, ScoreConfig(f_beta=beta))
    assert isinstance(result, Ok)
    assert _scores(result.value)["f_beta"] == 0.0


def test_column_vectors_and_invalid_inputs() -> None:
    """Only aligned binary vectors or N-by-one columns are accepted."""
    result = score_classifier(np.array([[1.0], [-1.0]]), np.array([[1.0], [0.0]]))
    assert isinstance(result, Ok)
    for logits, labels in (
        ([], []),
        ([0.0], [0.0, 1.0]),
        ([float("nan")], [1.0]),
        ([float("inf")], [0.0]),
        ([0.0], [0.5]),
        ([0.0], [-1.0]),
        ([True], [1.0]),
        (["0"], [1.0]),
        ([[0.0, 1.0]], [[0.0, 1.0]]),
    ):
        assert isinstance(score_classifier(logits, labels), Err)


def test_training_prevalence_baseline_and_probability_endpoints() -> None:
    """A fixed training probability is not fitted to evaluation labels."""
    baseline = score_classifier_baseline(0.25, [1.0, 0.0, 0.0, 0.0])
    impossible = score_classifier_baseline(0.0, [1.0, 0.0])
    assert isinstance(baseline, Ok) and isinstance(impossible, Ok)
    scores = _scores(baseline.value)
    assert scores["average_precision"] == 0.25
    assert scores["roc_auc"] == 0.5
    assert scores["brier_score"] == pytest.approx(3 / 16)
    assert scores["binary_cross_entropy"] == pytest.approx(
        -(math.log(0.25) + 3 * math.log(0.75)) / 4
    )
    assert _scores(impossible.value)["binary_cross_entropy"] is None
    assert impossible.value.losses == (None, 0.0)
    assert isinstance(score_classifier_baseline(1.1, [0.0]), Err)
