"""Pure probability calibration evidence with explicit empty-bin support.

Brier is an unweighted probability score. Equal-width reliability bins are
left-closed/right-open except the final closed bin. ECE depends on this binning.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from flight.libs.types import Err, Ok, Result

from tools.ml_models.analysis.contracts import CurveEvidence, MetricSupport, MetricValue, NamedCount
from tools.ml_models.analysis.metrics.inputs import binary_vectors, finite_mean


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    """One probability bin, with null measurements when its support is zero."""

    lower: float
    upper: float
    n: int
    mean_probability: float | None
    positive_fraction: float | None


@dataclass(frozen=True, slots=True)
class CalibrationEvidence:
    """Probability scores, exact bin aggregates, and display-ready coordinates."""

    metrics: tuple[MetricValue, ...]
    bins: tuple[ReliabilityBin, ...]
    curves: tuple[CurveEvidence, ...]
    support: MetricSupport


def score_calibration(
    probabilities: npt.ArrayLike,
    labels: npt.ArrayLike,
    *,
    n_bins: int = 10,
) -> Result[CalibrationEvidence, str]:
    """Measure Brier, reliability and ECE on aligned binary observations."""
    if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins < 2:
        return Err("calibration requires an integer bin count of at least two")
    validated = binary_vectors(probabilities, labels, probabilities=True)
    if isinstance(validated, Err):
        return Err(validated.error)
    probability, target = validated.value
    n = len(probability)
    n_positive = int(target.sum())
    support = MetricSupport(
        unit="IMAGE",
        n=n,
        counts=(
            NamedCount(name="n_positive", value=n_positive),
            NamedCount(name="n_negative", value=n - n_positive),
        ),
    )
    edges = np.arange(n_bins + 1, dtype=np.float64) / n_bins
    indices = np.minimum(np.searchsorted(edges, probability, side="right") - 1, n_bins - 1)
    bins: list[ReliabilityBin] = []
    for index in range(n_bins):
        selected = indices == index
        count = int(selected.sum())
        bins.append(
            ReliabilityBin(
                lower=float(edges[index]),
                upper=float(edges[index + 1]),
                n=count,
                mean_probability=finite_mean(probability[selected]) if count else None,
                positive_fraction=float(target[selected].mean()) if count else None,
            )
        )
    brier = finite_mean((probability - target.astype(np.float64)) ** 2)
    ece = math.fsum(
        bin.n / n * abs(bin.mean_probability - bin.positive_fraction)
        for bin in bins
        if bin.mean_probability is not None and bin.positive_fraction is not None
    )
    metrics = (
        MetricValue(
            name="brier_score",
            value=brier,
            status="AVAILABLE",
            support=support,
            aggregation="unweighted_probability_mean",
        ),
        MetricValue(
            name="expected_calibration_error",
            value=ece,
            status="AVAILABLE",
            support=support,
            aggregation="support_weighted_equal_width_bins",
        ),
    )
    reliability = CurveEvidence(
        name="calibration_reliability",
        x_name="mean_predicted_probability",
        y_name="observed_positive_fraction",
        x=tuple(
            bin.mean_probability
            if bin.mean_probability is not None
            else (bin.lower + bin.upper) / 2
            for bin in bins
        ),
        y=tuple(bin.positive_fraction for bin in bins),
        x_unit="probability",
        y_unit="fraction",
        support=support,
        notes=("Empty bins have missing outcomes; their x coordinate is the bin midpoint.",),
    )
    bin_support = CurveEvidence(
        name="calibration_support",
        x_name="probability_bin_midpoint",
        y_name="n_observations",
        x=tuple((bin.lower + bin.upper) / 2 for bin in bins),
        y=tuple(float(bin.n) for bin in bins),
        x_unit="probability",
        y_unit="count",
        support=support,
    )
    return Ok(CalibrationEvidence(metrics, tuple(bins), (reliability, bin_support), support))
