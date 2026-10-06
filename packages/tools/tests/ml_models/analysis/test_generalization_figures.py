"""Frozen stratum support, interval endpoints and heatmap missingness references."""

from dataclasses import replace

from flight.libs.types import Err, Ok
from tools.ml_models.analysis.config import GeneralizationConfig, ScoreConfig
from tools.ml_models.analysis.contracts import (
    AvailabilityRecord,
    ConfidenceInterval,
    CurveEvidence,
    MetricSupport,
    MetricValue,
    SplitEvidence,
    StratumEvidence,
)
from tools.ml_models.analysis.generalization_figures import generalization_figure_data
from tools.ml_models.analysis.metrics.generalization import GeneralizationEvidence


def _fixture() -> tuple[SplitEvidence, GeneralizationEvidence]:
    metric = MetricValue(
        name="recall",
        value=0.5,
        status="AVAILABLE",
        support=MetricSupport(unit="IMAGE", n=2),
        interval=ConfidenceInterval(
            lower=0.6,
            upper=0.8,
            confidence=0.95,
            method="percentile_recorded_group_bootstrap",
            n_replicates=10,
            n_valid=8,
            seed=3,
        ),
    )
    strata = (
        StratumEvidence("condition:cloud", None, (metric,), MetricSupport(unit="IMAGE", n=2)),
        StratumEvidence(
            "condition:cloud",
            "(recorded metadata missing)",
            (replace(metric, value=0.25, interval=None),),
            MetricSupport(unit="IMAGE", n=2),
        ),
        StratumEvidence(
            "gsd_lateral_by_truth_image_area_px",
            "[1,2)|[0,15)",
            (metric,),
            MetricSupport(unit="IMAGE", n=2),
        ),
        StratumEvidence(
            "gsd_lateral_by_truth_image_area_px",
            "[2,3)|[15,64)",
            (
                replace(
                    metric,
                    value=None,
                    status="UNAVAILABLE",
                    reason="No positive truth",
                    interval=None,
                ),
            ),
            MetricSupport(unit="IMAGE", n=2),
        ),
    )
    evidence = SplitEvidence(
        task="segmentor",
        split="val",
        dataset_hash="a" * 64,
        checkpoint_hash="c" * 64,
        support=MetricSupport(unit="IMAGE", n=4),
    )
    frozen = GeneralizationEvidence(
        evidence.dataset_hash,
        None,
        "segmentor",
        "val",
        ScoreConfig(),
        GeneralizationConfig(bootstrap_replicates=10),
        (metric,),
        (),
        strata,
        (),
        None,
        (replace(metric, value=0.0, interval=None),),
        (),
        (AvailabilityRecord(name="dates", status="UNAVAILABLE", reason="No recorded dates"),),
        ("Grouped holdouts do not prove temporal extrapolation.",),
    )
    return evidence, frozen


def test_frozen_interval_can_exclude_point_estimate_without_recentering() -> None:
    evidence, frozen = _fixture()
    result = generalization_figure_data(evidence, frozen)
    assert isinstance(result, Ok)
    whole = next(figure for figure in result.value if figure.identifier == "grouped_metric_recall")
    assert whole.series[0].y == (0.5,)
    assert whole.series[0].lower == (0.6,) and whole.series[0].upper == (0.8,)
    condition = next(figure for figure in result.value if figure.x_label == "condition:cloud")
    assert condition.series[0].point_support == (2, 2)
    assert condition.series[0].lower == (0.6, None)
    assert condition.x_categories[0] != condition.x_categories[1]


def test_heatmap_absent_and_unavailable_cells_stay_null_with_exact_support() -> None:
    evidence, frozen = _fixture()
    result = generalization_figure_data(evidence, frozen)
    assert isinstance(result, Ok)
    heatmap = next(figure for figure in result.value if figure.identifier.startswith("heatmap_"))
    assert heatmap.x_categories == ("[1,2)", "[2,3)")
    assert heatmap.y_categories == ("[0,15)", "[15,64)")
    assert heatmap.matrix == ((0.5, None), (None, None))
    assert heatmap.matrix_support == ((2, 0), (0, 2))


def test_baseline_and_missing_metadata_are_indexed_without_resampling() -> None:
    evidence, frozen = _fixture()
    result = generalization_figure_data(evidence, frozen)
    assert isinstance(result, Ok)
    baseline = next(figure for figure in result.value if figure.identifier == "baseline_recall")
    assert baseline.series[0].y == (0.5, 0.0)
    assert baseline.series[0].point_support == (2, 2)
    missing = next(figure for figure in result.value if figure.identifier.startswith("coverage_"))
    assert missing.reason == "No recorded dates"
    assert baseline.identity.checkpoint_hash == "c" * 64
    assert frozen.metrics[0].interval is not None


def test_mismatched_identity_and_duplicate_heatmap_cells_fail_closed() -> None:
    evidence, frozen = _fixture()
    assert isinstance(
        generalization_figure_data(replace(evidence, dataset_hash="d" * 64), frozen), Err
    )
    assert isinstance(generalization_figure_data(replace(evidence, split="test"), frozen), Err)
    duplicate = replace(frozen, strata=(*frozen.strata, frozen.strata[2]))
    assert isinstance(generalization_figure_data(evidence, duplicate), Err)


def test_stratum_pages_cover_every_frozen_cohort_without_sampling() -> None:
    evidence, frozen = _fixture()
    strata = tuple(
        StratumEvidence(
            "condition:cloud",
            str(index),
            (replace(frozen.metrics[0], value=index / 10),),
            MetricSupport(unit="IMAGE", n=2),
        )
        for index in range(9)
    )
    result = generalization_figure_data(evidence, replace(frozen, strata=strata))
    assert isinstance(result, Ok)
    pages = tuple(figure for figure in result.value if figure.identifier.startswith("stratum_"))
    assert [len(page.x_categories) for page in pages] == [6, 3]
    assert tuple(value for page in pages for value in page.series[0].y) == tuple(
        index / 10 for index in range(9)
    )
    assert tuple(label for page in pages for label in page.x_categories) == tuple(
        repr(str(index)) for index in range(9)
    )
    assert all(page.series[0].point_support == (2,) * len(page.x_categories) for page in pages)


def test_heatmap_tiles_preserve_all_cells_and_share_frozen_color_bounds() -> None:
    evidence, frozen = _fixture()
    strata = tuple(
        StratumEvidence(
            "gsd_lateral_by_truth_image_area_px",
            f"[{index},{index + 1})|[0,15)",
            (replace(frozen.metrics[0], value=index / 10),),
            MetricSupport(unit="IMAGE", n=2),
        )
        for index in range(9)
    )
    result = generalization_figure_data(evidence, replace(frozen, strata=strata))
    assert isinstance(result, Ok)
    tiles = tuple(figure for figure in result.value if figure.identifier.startswith("heatmap_"))
    assert [len(tile.x_categories) for tile in tiles] == [6, 3]
    assert tuple(value for tile in tiles for value in tile.matrix[0]) == tuple(
        index / 10 for index in range(9)
    )
    assert all(tile.matrix_range == (0.0, 0.8) for tile in tiles)
    assert tuple(count for tile in tiles for count in tile.matrix_support[0]) == (2,) * 9


def test_baseline_curve_unavailability_survives_recipe_reconstruction() -> None:
    evidence, frozen = _fixture()
    missing = CurveEvidence(
        name="threshold_recall",
        x_name="threshold",
        y_name="recall",
        x=(0.0, 1.0),
        y=(None, None),
        x_unit="probability",
        y_unit="fraction",
        support=MetricSupport(unit="IMAGE", n=2),
    )
    result = generalization_figure_data(evidence, replace(frozen, baseline_curves=(missing,)))
    assert isinstance(result, Ok)
    figure = next(
        figure for figure in result.value if figure.identifier == "baseline_curve_threshold_recall"
    )
    assert figure.reason == "No eligible captured values for this curve"
    assert figure.series[0].y == (None, None)
