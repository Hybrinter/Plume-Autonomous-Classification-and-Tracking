"""Independent group-bootstrap, captured strata and frozen-baseline references."""

from dataclasses import replace

import numpy as np
import pytest
from flight.libs.types import Err, Ok, Result
from tools.ml_models.analysis.capture import CaptureRow
from tools.ml_models.analysis.config import GeneralizationConfig, ScoreConfig
from tools.ml_models.analysis.contracts import (
    MetricSupport,
    MetricValue,
    SampleKey,
    Split,
)
from tools.ml_models.analysis.metrics.definitions import metric_definition
from tools.ml_models.analysis.metrics.generalization import (
    BootstrapEvidence,
    development_gaps,
    fit_baseline,
    group_bootstrap,
    measure_generalization,
    score_captured_rows,
)
from tools.ml_models.analysis.metrics.segmentation import (
    aggregate_segmentation,
    score_segmentation_image,
)
from tools.ml_models.analysis.metrics.spatial import score_spatial
from tools.ml_models.dataset.raw import ConditionTag, ObservationMetadata


def _row(index: int, group: str, label: int, logit: float, *, split: Split = "val") -> CaptureRow:
    return CaptureRow(
        key=SampleKey(
            dataset_hash="d" * 64,
            task="classifier",
            split=split,
            spatial_shard=(2, 3),
            row_index=index,
            tile_id=f"tile-{index}",
            element="id",
        ),
        group_id=group,
        bin_id="",
        label=label,
        gsd_m=(2.0, 3.0),
        metrics=(
            MetricValue(
                name="logit",
                value=logit,
                status="AVAILABLE",
                support=MetricSupport(unit="IMAGE", n=1),
            ),
        ),
        failure_score=0,
    )


def test_group_bootstrap_matches_seeded_whole_group_reference_not_row_bootstrap() -> None:
    rows = (
        _row(0, "a", 1, -4.0),
        _row(1, "a", 1, -4.0),
        _row(2, "a", 1, -4.0),
        _row(3, "b", 0, -4.0),
    )
    cfg = GeneralizationConfig(bootstrap_replicates=40, seed=7, confidence=0.8)
    result = group_bootstrap(rows, ScoreConfig(), cfg)
    assert isinstance(result, Ok)
    rng = np.random.default_rng(7)
    expected = []
    for _ in range(40):
        draw = rng.integers(0, 2, size=2)
        a, b = int(np.count_nonzero(draw == 0)), int(np.count_nonzero(draw == 1))
        expected.append(b / (3 * a + b))
    accuracy = next(m for m in result.value.metrics if m.name == "accuracy")
    assert accuracy.value == 0.25
    assert accuracy.interval is not None
    assert accuracy.interval.lower == np.quantile(expected, 0.1, method="linear")
    assert accuracy.interval.upper == np.quantile(expected, 0.9, method="linear")
    assert accuracy.interval.n_valid == 40
    assert accuracy.interval.method == "percentile_recorded_group_bootstrap"
    reversed_result = group_bootstrap(tuple(reversed(rows)), ScoreConfig(), cfg)
    assert isinstance(reversed_result, Ok)
    assert result.value == reversed_result.value


def test_missing_class_replicates_are_counted_and_few_groups_are_unavailable() -> None:
    rows = (_row(0, "a", 1, 4.0), _row(1, "b", 0, -4.0))
    cfg = GeneralizationConfig(bootstrap_replicates=40, seed=9)
    result = group_bootstrap(rows, ScoreConfig(), cfg)
    assert isinstance(result, Ok)
    rng = np.random.default_rng(9)
    valid = sum(len(set(rng.integers(0, 2, size=2))) == 2 for _ in range(40))
    roc = next(m for m in result.value.metrics if m.name == "roc_auc")
    assert roc.interval is not None
    assert roc.interval.n_valid == valid < 40
    audit = next(a for a in result.value.intervals if a.metric == "roc_auc")
    assert audit.n_valid == valid and audit.n_invalid == 40 - valid
    single = group_bootstrap(rows[:1], ScoreConfig(), cfg)
    assert isinstance(single, Ok)
    assert all(m.interval is None for m in single.value.metrics)
    assert all(a.reason for a in single.value.intervals)


def test_strata_use_recorded_categories_and_missing_is_not_a_literal_tag() -> None:
    rows = (
        replace(
            _row(0, "a", 1, 2.0),
            metadata=ObservationMetadata(
                observation_id="o-a",
                acquired_at_utc="2026-01-02T03:04:05Z",
                conditions=(ConditionTag(name="sky", value="missing"),),
            ),
            gsd_nominal=True,
        ),
        replace(_row(1, "b", 0, -2.0), gsd_m=(4.0, 6.0), gsd_nominal=False),
    )
    result = measure_generalization(
        rows,
        ScoreConfig(),
        GeneralizationConfig(bootstrap_replicates=8, gsd_edges_m=(0.0, 3.0, 5.0)),
    )
    assert isinstance(result, Ok)
    strata = {(s.name, s.value): s for s in result.value.strata}
    assert ("condition:sky", "missing") in strata
    assert ("condition:sky", None) in strata
    assert strata["condition:sky", "missing"].support.n == 1
    assert strata["condition:sky", None].support.n == 1
    assert ("acquisition_month_utc", "2026-01") in strata
    assert ("acquisition_month_utc", None) in strata
    assert ("gsd_provenance", "nominal") in strata
    assert ("gsd_provenance", "recorded_non_nominal") in strata
    assert strata["truth_image_area_px", None].support.n == 2
    assert any(
        output.reason for output in result.value.outputs if output.name == "classifier_truth_size"
    )


def test_baseline_is_fit_once_on_canonical_train_never_held_out() -> None:
    train = (_row(0, "a", 1, 4.0, split="train"), _row(1, "b", 0, -4.0, split="train"))
    fitted = fit_baseline(train)
    assert isinstance(fitted, Ok)
    assert fitted.value.training_prevalence == 0.5
    assert fitted.value.n_images == 2
    held_out = (_row(0, "c", 1, 4.0, split="test"), _row(1, "d", 1, 4.0, split="test"))
    assert isinstance(fit_baseline(held_out), Err)
    result = measure_generalization(
        held_out,
        ScoreConfig(),
        GeneralizationConfig(bootstrap_replicates=8),
        baseline=fitted.value,
    )
    assert isinstance(result, Ok)
    baseline = {m.name: m for m in result.value.baseline_metrics}
    assert baseline["brier_score"].value == 0.25
    assert baseline["binary_cross_entropy"].value == pytest.approx(np.log(2))
    assert fitted.value.training_prevalence == 0.5


def test_duplicate_capture_identity_and_leaking_recorded_observation_fail_closed() -> None:
    a, b = _row(0, "a", 1, 2.0), _row(1, "b", 0, -2.0)
    cfg = GeneralizationConfig(bootstrap_replicates=8)
    assert isinstance(measure_generalization((a, a), ScoreConfig(), cfg), Err)
    metadata = ObservationMetadata(observation_id="shared")
    assert isinstance(
        measure_generalization(
            (replace(a, metadata=metadata), replace(b, metadata=metadata)),
            ScoreConfig(),
            cfg,
        ),
        Err,
    )
    assert isinstance(
        score_captured_rows(
            (replace(a, metrics=()), b),
            ScoreConfig(),
        ),
        Err,
    )


def _segmented_row(index: int, truth: np.ndarray, predicted: np.ndarray, group: str) -> CaptureRow:
    cfg = ScoreConfig(min_blob_area_px=2, pixel_histogram_bins=8)
    label = float(bool(truth.any()))
    logits = np.where(predicted, 3.0, -3.0)
    measured = score_segmentation_image(
        logits, truth, label=label, verified_empty=not bool(label), gsd=(2.0, 3.0), cfg=cfg
    )
    spatial = score_spatial(logits, truth, gsd=(2.0, 3.0), cfg=cfg)
    assert isinstance(measured, Ok) and isinstance(spatial, Ok)
    row = measured.value
    fields = (
        ("foreground_iou", row.iou),
        ("foreground_dice", row.dice),
        ("binary_cross_entropy", row.bce),
        ("brier_score", row.brier),
        ("target_area_px", row.target_area_px),
        ("predicted_area_px", row.predicted_area_px),
        ("target_area_m2", row.target_area_m2),
        ("predicted_area_m2", row.predicted_area_m2),
        ("true_positive_pixels", row.tp),
        ("false_positive_pixels", row.fp),
        ("false_negative_pixels", row.fn),
        ("predicted_blobs", row.n_predicted_blobs),
    )
    return CaptureRow(
        key=SampleKey(
            dataset_hash="d" * 64,
            task="segmentor",
            split="val",
            spatial_shard=truth.shape,
            row_index=index,
            tile_id=f"tile-{index}",
            element="id",
        ),
        group_id=group,
        bin_id="",
        label=label,
        gsd_m=(2.0, 3.0),
        spatial=spatial.value,
        metrics=tuple(
            MetricValue(
                name=name,
                value=value,
                status="AVAILABLE" if value is not None else "UNAVAILABLE",
                reason="missing GSD" if value is None else None,
                support=MetricSupport(unit="IMAGE", n=1),
            )
            for name, value in fields
        ),
        failure_score=0,
    )


def test_segmentation_captured_reductions_match_shared_authority_and_definitions() -> None:
    masks = (
        np.ones((2, 3), dtype=np.uint8),
        np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0]], dtype=np.uint8),
        np.zeros((2, 2), dtype=np.uint8),
    )
    predictions = (masks[0].copy(), np.zeros_like(masks[1]), np.zeros_like(masks[2]))
    rows = tuple(
        _segmented_row(i, truth, predicted, str(i))
        for i, (truth, predicted) in enumerate(zip(masks, predictions, strict=True))
    )
    cfg = ScoreConfig(min_blob_area_px=2, pixel_histogram_bins=8)
    originals = tuple(
        score_segmentation_image(
            np.where(prediction, 3.0, -3.0),
            truth,
            label=float(bool(truth.any())),
            verified_empty=not bool(truth.any()),
            gsd=(2.0, 3.0),
            cfg=cfg,
        )
        for truth, prediction in zip(masks, predictions, strict=True)
    )
    assert all(isinstance(original, Ok) for original in originals)
    known = aggregate_segmentation(
        tuple(original.value for original in originals if isinstance(original, Ok))
    )
    captured = score_captured_rows(rows, cfg)
    assert isinstance(known, Ok) and isinstance(captured, Ok)
    expected = {metric.name: metric for metric in known.value.metrics}
    for metric in captured.value:
        definition = metric_definition(metric.name)
        assert isinstance(definition, Ok), metric.name
        assert definition.value.unit == metric.unit
        assert definition.value.aggregation == metric.aggregation
        if metric.name in expected:
            target = expected[metric.name]
            assert (
                metric.value == pytest.approx(target.value)
                if target.value is not None
                else metric.value is None
            )
            assert metric.support.unit == target.support.unit
            assert metric.support.n == target.support.n
    frozen = measure_generalization(rows, cfg, GeneralizationConfig(bootstrap_replicates=8))
    assert isinstance(frozen, Ok)
    assert frozen.value.strata


def test_gsd_identity_labels_do_not_round_distinct_recorded_geometries_together() -> None:
    rows = (
        replace(_row(0, "a", 1, 2.0), gsd_m=(2.1234561, 3.0)),
        replace(_row(1, "b", 0, -2.0), gsd_m=(2.1234562, 3.0)),
    )
    result = measure_generalization(
        rows, ScoreConfig(), GeneralizationConfig(bootstrap_replicates=8)
    )
    assert isinstance(result, Ok)
    pairs = [stratum for stratum in result.value.strata if stratum.name == "gsd_pair_m"]
    assert len(pairs) == 2
    assert all(stratum.support.n == 1 for stratum in pairs)


def test_truth_component_size_strata_keep_small_misses_separate_from_image_area() -> None:
    truth = np.zeros((4, 6), dtype=np.uint8)
    truth[:2, :3] = 1
    truth[3, 5] = 1
    prediction = truth.copy()
    prediction[3, 5] = 0
    rows = tuple(_segmented_row(i, truth, prediction, str(i)) for i in range(2))
    result = measure_generalization(
        rows,
        ScoreConfig(min_blob_area_px=2, pixel_histogram_bins=8),
        GeneralizationConfig(bootstrap_replicates=8, size_edges_px=(0.0, 2.0, 6.0)),
    )
    assert isinstance(result, Ok)
    strata = {(s.name, s.value): s for s in result.value.strata}
    tiny = strata["truth_component_area_px", "[0,2)"]
    large = strata["truth_component_area_px", ">=6"]
    assert tiny.support.unit == large.support.unit == "COMPONENT"
    assert tiny.support.n == large.support.n == 2
    metrics = {m.name: m for m in tiny.metrics}
    assert metrics["component_recall"].value == 0
    assert metrics["component_recall"].interval is not None
    assert (
        metrics["component_recall"].interval.lower
        == metrics["component_recall"].interval.upper
        == 0
    )
    assert metrics["component_precision"].value is None
    assert metrics["matched_centroid_error_px_mean"].value is None
    assert next(m for m in large.metrics if m.name == "component_recall").value == 1
    assert strata["truth_image_area_px", ">=6"].support.unit == "IMAGE"


def test_insufficient_groups_do_not_claim_available_intervals() -> None:
    result = measure_generalization(
        (_row(0, "only", 1, 2.0),),
        ScoreConfig(),
        GeneralizationConfig(bootstrap_replicates=8),
    )
    assert isinstance(result, Ok)
    output = next(o for o in result.value.outputs if o.name == "grouped_intervals")
    assert output.status == "UNAVAILABLE" and output.reason
    assert result.value.dataset_hash == "d" * 64
    assert result.value.task == "classifier" and result.value.split == "val"
    assert result.value.dataset_manifest_hash is None
    assert result.value.score_config == ScoreConfig()
    assert result.value.generalization_config == GeneralizationConfig(bootstrap_replicates=8)


def test_development_gap_retains_support_and_refuses_test_or_cross_dataset() -> None:
    train = (_row(0, "a", 1, 4.0, split="train"), _row(1, "b", 0, -4.0, split="train"))
    validation = (_row(0, "c", 1, -4.0), _row(1, "d", 0, -4.0))
    result = development_gaps(train, validation, ScoreConfig(), checkpoint_hash="c" * 64)
    assert isinstance(result, Ok)
    accuracy = next(gap for gap in result.value.gaps if gap.name == "accuracy")
    assert accuracy.difference == 0.5
    assert accuracy.train.support.n == accuracy.validation.support.n == 2
    assert result.value.checkpoint_hash == "c" * 64
    assert isinstance(
        development_gaps(
            train,
            tuple(replace(row, key=replace(row.key, split="test")) for row in validation),
            ScoreConfig(),
            checkpoint_hash="c" * 64,
        ),
        Err,
    )
    assert isinstance(
        development_gaps(
            train,
            tuple(replace(row, key=replace(row.key, dataset_hash="b" * 64)) for row in validation),
            ScoreConfig(),
            checkpoint_hash="c" * 64,
        ),
        Err,
    )


def test_augmentation_duplicates_cannot_weight_captured_statistics() -> None:
    row = _row(0, "a", 1, 4.0)
    duplicate = replace(row, key=replace(row.key, row_index=1, element="flip_h"))
    assert isinstance(score_captured_rows((row, duplicate), ScoreConfig()), Err)


def test_frozen_mask_counts_cannot_be_relabelled_with_changed_scoring_settings() -> None:
    mask = np.ones((2, 3), dtype=np.uint8)
    row = _segmented_row(0, mask, mask, "group")
    assert isinstance(
        score_captured_rows(
            (row,),
            ScoreConfig(min_blob_area_px=2, mask_probability_threshold=0.9),
        ),
        Err,
    )
    bad = replace(
        row,
        metrics=tuple(
            replace(metric, value=1.0) if metric.name == "target_area_px" else metric
            for metric in row.metrics
        ),
    )
    assert isinstance(
        score_captured_rows(
            (bad,),
            ScoreConfig(min_blob_area_px=2, pixel_histogram_bins=8),
        ),
        Err,
    )


def test_identical_named_cohorts_reuse_the_frozen_bootstrap_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.ml_models.analysis.metrics.generalization as module

    rows = (_row(0, "a", 1, 2.0), _row(1, "b", 0, -2.0))
    original = module.group_bootstrap
    calls: list[tuple[SampleKey, ...]] = []

    def counted(
        cohort: tuple[CaptureRow, ...],
        score: ScoreConfig,
        cfg: GeneralizationConfig,
    ) -> Result[BootstrapEvidence, str]:
        calls.append(tuple(row.key for row in cohort))
        return original(cohort, score, cfg)

    monkeypatch.setattr(module, "group_bootstrap", counted)
    result = measure_generalization(
        rows, ScoreConfig(), GeneralizationConfig(bootstrap_replicates=8)
    )
    assert isinstance(result, Ok)
    assert len(calls) == 1
    assert result.value.strata
