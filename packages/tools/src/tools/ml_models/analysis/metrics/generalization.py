"""Frozen captured-row strata, train-only baselines and recorded-group bootstrap.

Whole groups, not pixels or rows, are resampled with related variants intact.
Group IDs do not establish statistical independence. Unsupported original
metadata and compact-capture detail remain unavailable. Every derived interval
and stratum is persisted by callers once; renderers never resample it.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, replace

import numpy as np
from flight.libs.types import Err, Ok, Result

from tools.ml_models.analysis.capture import CaptureRow
from tools.ml_models.analysis.config import GeneralizationConfig, ScoreConfig
from tools.ml_models.analysis.contracts import (
    AvailabilityRecord,
    ConfidenceInterval,
    CurveEvidence,
    MetricSupport,
    MetricValue,
    NamedCount,
    SampleKey,
    Split,
    StratumEvidence,
    Task,
    is_sha256,
)
from tools.ml_models.analysis.metrics.classifier import score_classifier, score_classifier_baseline
from tools.ml_models.analysis.metrics.definitions import (
    MetricDefinition,
    metric_definition,
)
from tools.ml_models.analysis.metrics.spatial import aggregate_spatial


@dataclass(frozen=True, slots=True)
class IntervalAudit:
    """Declared replicate budget, actual valid/invalid support and explicit unavailability."""

    metric: str
    n_replicates: int
    n_attempted: int
    n_valid: int
    n_invalid: int
    n_groups: int
    reason: str | None


@dataclass(frozen=True, slots=True)
class BootstrapEvidence:
    """Point metrics and intervals plus the replicate audit; no replicate arrays retained."""

    metrics: tuple[MetricValue, ...]
    intervals: tuple[IntervalAudit, ...]


@dataclass(frozen=True, slots=True)
class BaselineSpec:
    """Fixed train-only constant classifier probability or all-background segmentation policy."""

    task: str
    training_dataset_hash: str
    n_images: int
    n_groups: int
    training_prevalence: float | None
    method: str = "fixed_canonical_train_prevalence_or_all_background"


@dataclass(frozen=True, slots=True)
class StratumIntervals:
    """Audit keyed to the same named nullable cohort as its stratum metrics."""

    name: str
    value: str | None
    intervals: tuple[IntervalAudit, ...]


@dataclass(frozen=True, slots=True)
class GeneralizationEvidence:
    """Frozen cohort/stratum intervals, fixed baseline comparisons and explicit coverage caveats."""

    dataset_hash: str
    dataset_manifest_hash: str | None
    task: Task
    split: Split
    score_config: ScoreConfig
    generalization_config: GeneralizationConfig
    metrics: tuple[MetricValue, ...]
    intervals: tuple[IntervalAudit, ...]
    strata: tuple[StratumEvidence, ...]
    stratum_intervals: tuple[StratumIntervals, ...]
    baseline: BaselineSpec | None
    baseline_metrics: tuple[MetricValue, ...]
    baseline_curves: tuple[CurveEvidence, ...]
    outputs: tuple[AvailabilityRecord, ...]
    warnings: tuple[str, ...]
    method: str = "exact_captured_cohorts_seeded_recorded_group_bootstrap_v1"


@dataclass(frozen=True, slots=True)
class MetricReference:
    """Known metric definition or explicit unknown-name state; no inferred formula/direction."""

    name: str
    definition: MetricDefinition | None
    reason: str | None


@dataclass(frozen=True, slots=True)
class ComponentObservation:
    """One truth component joined to its parent capture/group, never a new independent sample."""

    row: CaptureRow
    truth_id: int


@dataclass(frozen=True, slots=True)
class DevelopmentGap:
    """Explicit train-minus-validation scalar difference with both populations retained."""

    name: str
    train: MetricValue
    validation: MetricValue
    difference: float | None
    reason: str | None


@dataclass(frozen=True, slots=True)
class DevelopmentEvidence:
    """Comparable dataset/scoring populations bound by the caller to one selected checkpoint."""

    dataset_hash: str
    checkpoint_hash: str
    gaps: tuple[DevelopmentGap, ...]
    dataset_manifest_hash: str | None
    score_config: ScoreConfig
    task: Task
    method: str = "same_checkpoint_same_dataset_train_minus_validation"


def _ordered(rows: tuple[CaptureRow, ...]) -> tuple[CaptureRow, ...]:
    """Use source keys for stable traversal independent of incoming row order."""
    return tuple(
        sorted(
            rows,
            key=lambda r: (
                r.group_id,
                r.key.tile_id,
                r.key.spatial_shard,
                r.key.row_index,
                r.key.element,
            ),
        )
    )


def _validate(rows: tuple[CaptureRow, ...]) -> Result[None, str]:
    """Require one captured task/split/dataset cohort and nonconflicting recorded identities."""
    if not rows:
        return Err("generalization requires captured rows")
    first = rows[0].key
    identities = {
        (r.key.task, r.key.split, r.key.dataset_hash, r.dataset_manifest_hash) for r in rows
    }
    if identities != {(first.task, first.split, first.dataset_hash, rows[0].dataset_manifest_hash)}:
        return Err("generalization rows mix task, split or dataset identities")
    canonical = {(r.key.tile_id, r.bin_id, r.gsd_m, r.key.spatial_shard) for r in rows}
    if (
        len({r.key for r in rows}) != len(rows)
        or len(canonical) != len(rows)
        or any(not r.group_id.strip() for r in rows)
    ):
        return Err("generalization needs unique captured keys and recorded nonblank groups")
    sources: dict[str, tuple[object, ...]] = {}
    for row in rows:
        observation = row.metadata.observation_id
        if observation is not None:
            source = (row.group_id, row.label, row.metadata)
            if sources.setdefault(observation, source) != source:
                return Err("recorded observation has conflicting group, label or metadata")
    return Ok(None)


def _metric(
    name: str,
    value: float | None,
    n: int,
    *,
    aggregation: str,
    unit: str = "dimensionless",
    reason: str = "no eligible captured support",
) -> MetricValue:
    """Declare the exact captured population, reduction and undefined state."""
    return MetricValue(
        name=name,
        value=value,
        status="AVAILABLE" if value is not None else "UNAVAILABLE",
        reason=None if value is not None else reason,
        unit=unit,
        aggregation=aggregation,
        support=MetricSupport(unit="IMAGE", n=n),
    )


def _values(row: CaptureRow) -> dict[str, float | None]:
    """Read available captured values without manufacturing missing measurements."""
    return {metric.name: metric.value for metric in row.metrics}


def _score(
    rows: tuple[CaptureRow, ...],
    score: ScoreConfig,
) -> Result[tuple[MetricValue, ...], str]:
    """Use shared classifier authority or exact explicit-mask sufficient-statistic reductions."""
    if rows[0].key.task == "classifier":
        logits = tuple(_values(row).get("logit") for row in rows)
        if any(logit is None for logit in logits):
            return Err("classifier generalization requires captured finite logits")
        resolved = replace(score, probability_thresholds=(score.classifier_probability_threshold,))
        measured = score_classifier(
            tuple(logit for logit in logits if logit is not None),
            tuple(row.label for row in rows),
            resolved,
        )
        if isinstance(measured, Err):
            return measured
        return Ok(measured.value.metrics)
    records = tuple(
        {name: value for name, value in _values(row).items() if value is not None} for row in rows
    )
    for row in rows:
        if row.spatial is None:
            return Err(
                "segmentor generalization requires frozen mask/blob/tolerance settings; "
                "re-analyze the checkpoint"
            )
        local, boundary = row.spatial.localization, row.spatial.boundary
        tolerance = (
            score.boundary_tolerance_m
            if score.boundary_tolerance_m is not None
            else score.boundary_tolerance_px
        )
        tolerance_unit = "m" if score.boundary_tolerance_m is not None else "pixel"
        if (local.blob_probability_threshold, local.min_blob_area_px, local.match_iou_min) != (
            score.blob_probability_threshold,
            score.min_blob_area_px,
            score.match_iou_min,
        ) or (boundary.mask_probability_threshold, boundary.tolerance, boundary.tolerance_unit) != (
            score.mask_probability_threshold,
            tolerance,
            tolerance_unit,
        ):
            return Err(
                "frozen segmentation scoring settings differ; "
                "re-analyze rather than relabel captured counts"
            )
    required = (
        "foreground_iou",
        "foreground_dice",
        "binary_cross_entropy",
        "brier_score",
        "target_area_px",
        "true_positive_pixels",
        "false_positive_pixels",
        "false_negative_pixels",
        "predicted_blobs",
        "predicted_area_px",
    )
    if any(any(record.get(name) is None for name in required) for record in records):
        return Err("segmentor generalization requires complete captured overlap/count/loss scalars")
    for record in records:
        if any(
            value is None or value < 0 or int(value) != value
            for name in required
            if name.endswith("_pixels") or name.endswith("_px") or name == "predicted_blobs"
            for value in (record[name],)
        ):
            return Err("captured segmentation counts must be exact nonnegative integers")
    for row, record in zip(rows, records, strict=True):
        pixels = row.key.spatial_shard[0] * row.key.spatial_shard[1]
        if (
            float(record["true_positive_pixels"]) + float(record["false_negative_pixels"])
            != record["target_area_px"]
            or float(record["true_positive_pixels"]) + float(record["false_positive_pixels"])
            != record["predicted_area_px"]
            or float(record["true_positive_pixels"])
            + float(record["false_positive_pixels"])
            + float(record["false_negative_pixels"])
            > pixels
            or row.label == 0
            and record["target_area_px"] != 0
        ):
            return Err("captured segmentation pixel counts contradict target/prediction support")
        if (
            row.spatial is not None
            and sum(component.area_px for component in row.spatial.localization.truth)
            != record["target_area_px"]
        ):
            return Err("captured scalar truth area conflicts with unfiltered component geometry")
    n = len(rows)
    positive = tuple(record for record in records if record["target_area_px"] != 0)
    verified_empty = tuple(
        record
        for row, record in zip(rows, records, strict=True)
        if row.label == 0 and record["target_area_px"] == 0
    )
    tp, fp, fn = (
        sum(int(record[name]) for record in records)
        for name in (
            "true_positive_pixels",
            "false_positive_pixels",
            "false_negative_pixels",
        )
    )
    metrics = tuple(
        _metric(name, value, support, aggregation=aggregation)
        for name, value, support, aggregation in (
            (
                "foreground_iou_mean_positive_images",
                math.fsum(float(record["foreground_iou"]) / len(positive) for record in positive)
                if positive
                else None,
                len(positive),
                "equal_image_mean",
            ),
            (
                "foreground_dice_mean_positive_images",
                math.fsum(float(record["foreground_dice"]) / len(positive) for record in positive)
                if positive
                else None,
                len(positive),
                "equal_image_mean",
            ),
            (
                "foreground_iou_mean_all_annotated_images",
                math.fsum(float(record["foreground_iou"]) / n for record in records),
                n,
                "equal_image_mean",
            ),
            (
                "foreground_dice_mean_all_annotated_images",
                math.fsum(float(record["foreground_dice"]) / n for record in records),
                n,
                "equal_image_mean",
            ),
            (
                "foreground_iou_global",
                tp / (tp + fp + fn) if tp + fp + fn else None,
                n,
                "pooled_pixel_counts",
            ),
            (
                "foreground_dice_global",
                2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                n,
                "pooled_pixel_counts",
            ),
            (
                "foreground_precision_global",
                tp / (tp + fp) if tp + fp else None,
                n,
                "pooled_pixel_counts",
            ),
            (
                "foreground_recall_global",
                tp / (tp + fn) if tp + fn else None,
                n,
                "pooled_pixel_counts",
            ),
            (
                "binary_cross_entropy_mean_images",
                math.fsum(float(record["binary_cross_entropy"]) / n for record in records),
                n,
                "equal_image_mean",
            ),
            (
                "brier_score_mean_images",
                math.fsum(float(record["brier_score"]) / n for record in records),
                n,
                "equal_image_mean",
            ),
            (
                "verified_negative_any_foreground_rate",
                sum(float(record["predicted_area_px"]) > 0 for record in verified_empty)
                / len(verified_empty)
                if verified_empty
                else None,
                len(verified_empty),
                "equal_image_mean",
            ),
            (
                "verified_negative_any_blob_rate",
                sum(float(record["predicted_blobs"]) > 0 for record in verified_empty)
                / len(verified_empty)
                if verified_empty
                else None,
                len(verified_empty),
                "equal_image_mean",
            ),
            (
                "verified_negative_false_blobs_mean",
                math.fsum(
                    float(record["predicted_blobs"]) / len(verified_empty)
                    for record in verified_empty
                )
                if verified_empty
                else None,
                len(verified_empty),
                "equal_image_mean",
            ),
        )
    )
    total_pixels = sum(row.key.spatial_shard[0] * row.key.spatial_shard[1] for row in rows)
    pixel_support = MetricSupport(
        unit="PIXEL", n=total_pixels, counts=(NamedCount(name="n_images", value=n),)
    )
    metrics = tuple(
        replace(metric, support=pixel_support, threshold=score.mask_probability_threshold)
        if metric.aggregation == "pooled_pixel_counts"
        else replace(metric, unit="fraction")
        if metric.name.startswith("verified_negative_any")
        else replace(metric, unit="blobs_per_image")
        if metric.name == "verified_negative_false_blobs_mean"
        else metric
        for metric in metrics
    )
    metrics = tuple(
        replace(metric, threshold=score.mask_probability_threshold)
        if metric.name.startswith(("foreground_iou", "foreground_dice"))
        else metric
        for metric in metrics
    )
    for source_name, name in (
        ("binary_cross_entropy", "binary_cross_entropy_mean_pixels"),
        ("brier_score", "brier_score_mean_pixels"),
    ):
        metrics += (
            replace(
                _metric(
                    name,
                    math.fsum(
                        float(record[source_name])
                        * (row.key.spatial_shard[0] * row.key.spatial_shard[1] / total_pixels)
                        for row, record in zip(rows, records, strict=True)
                    ),
                    n,
                    aggregation="pixel_weighted_image_means",
                ),
                support=pixel_support,
            ),
        )
    for name, absolute in (
        ("area_signed_error_mean_px", False),
        ("area_absolute_error_mean_px", True),
    ):
        differences = tuple(
            float(record["predicted_area_px"]) - float(record["target_area_px"])
            for record in records
        )
        metrics += (
            _metric(
                name,
                math.fsum((abs(value) if absolute else value) / n for value in differences),
                n,
                aggregation="equal_image_mean",
                unit="pixels",
            ),
        )
    spatial = tuple(row.spatial for row in rows if row.spatial is not None)
    spatial_evidence = aggregate_spatial(spatial)
    if isinstance(spatial_evidence, Err):
        return spatial_evidence
    metrics += spatial_evidence.value.metrics
    return Ok(metrics)


def score_captured_rows(
    rows: tuple[CaptureRow, ...],
    score: ScoreConfig,
) -> Result[tuple[MetricValue, ...], str]:
    """Compute exact supported scalar metrics from one identity-consistent captured cohort."""
    checked = _validate(rows)
    if isinstance(checked, Err):
        return checked
    try:
        return _score(_ordered(rows), score)
    except (ValueError, TypeError, OverflowError, RuntimeError) as exc:
        return Err(f"captured cohort reduction failed: {exc}")


def group_bootstrap(
    rows: tuple[CaptureRow, ...],
    score: ScoreConfig,
    cfg: GeneralizationConfig,
) -> Result[BootstrapEvidence, str]:
    """Freeze seeded percentile intervals by resampling whole recorded groups with replacement.

    Invalid class/support replicates are counted per metric, not imputed.
    Fewer than two recorded groups or two valid replicates leaves intervals
    unavailable. Group IDs need not identify statistically independent units;
    the grouped-random coverage design is not a temporal or unseen-GSD holdout.
    """
    checked = _validate(rows)
    if isinstance(checked, Err):
        return checked
    try:
        ordered = _ordered(rows)
        point = _score(ordered, score)
        if isinstance(point, Err):
            return point
        groups: dict[str, list[CaptureRow]] = defaultdict(list)
        for row in ordered:
            groups[row.group_id].append(row)
        cohorts = tuple(tuple(cohort) for _, cohort in sorted(groups.items()))
        estimates: dict[str, list[float]] = {
            metric.name: []
            for metric in point.value
            if metric.unit not in ("count", "component", "image")
            and not metric.aggregation.endswith("_count")
        }
        attempted = 0
        if len(cohorts) >= 2:
            rng = np.random.default_rng(cfg.seed)
            for _ in range(cfg.bootstrap_replicates):
                selected = rng.integers(0, len(cohorts), size=len(cohorts))
                replicate = tuple(row for index in selected for row in cohorts[int(index)])
                measured = _score(replicate, score)
                if isinstance(measured, Err):
                    return measured
                attempted += 1
                for metric in measured.value:
                    if metric.name in estimates and metric.value is not None:
                        estimates[metric.name].append(metric.value)
        metrics: list[MetricValue] = []
        audits: list[IntervalAudit] = []
        for metric in point.value:
            if metric.name not in estimates:
                metrics.append(metric)
                continue
            values = estimates[metric.name]
            reason = (
                "fewer than two recorded groups"
                if len(cohorts) < 2
                else "point metric unavailable"
                if metric.value is None
                else "fewer than two valid support/class replicates"
                if len(values) < 2
                else None
            )
            interval = None
            if reason is None:
                lower, upper = np.quantile(
                    values, ((1 - cfg.confidence) / 2, (1 + cfg.confidence) / 2), method="linear"
                )
                interval = ConfidenceInterval(
                    lower=float(lower),
                    upper=float(upper),
                    confidence=cfg.confidence,
                    method="percentile_recorded_group_bootstrap",
                    n_replicates=cfg.bootstrap_replicates,
                    n_valid=len(values),
                    seed=cfg.seed,
                )
            metrics.append(replace(metric, interval=interval))
            audits.append(
                IntervalAudit(
                    metric.name,
                    cfg.bootstrap_replicates,
                    attempted,
                    len(values),
                    attempted - len(values),
                    len(cohorts),
                    reason,
                )
            )
        return Ok(BootstrapEvidence(tuple(metrics), tuple(audits)))
    except (ValueError, TypeError, OverflowError, RuntimeError) as exc:
        return Err(f"group bootstrap failed: {exc}")


def _component_metrics(
    observations: tuple[ComponentObservation, ...],
) -> tuple[MetricValue, ...]:
    """Count detection of selected truth components; precision cannot assign spurious blobs to
    truth size."""
    n = len(observations)
    matches = tuple(
        match
        for observation in observations
        if observation.row.spatial is not None
        for match in observation.row.spatial.localization.matches
        if match.truth_id == observation.truth_id
    )
    support = MetricSupport(unit="COMPONENT", n=n)
    metrics: tuple[MetricValue, ...] = (
        replace(
            _metric(
                "component_recall",
                len(matches) / n if n else None,
                n,
                aggregation="pooled_truth_match_fraction",
            ),
            support=support,
        ),
        replace(
            _metric(
                "component_precision",
                None,
                0,
                aggregation="pooled_retained_prediction_match_fraction",
                reason=(
                    "unmatched prediction blobs cannot be assigned to a ground-truth-size stratum"
                ),
            ),
            support=MetricSupport(unit="COMPONENT", n=0),
        ),
    )
    for unit in ("px", "m"):
        values = tuple(
            value
            for match in matches
            if (value := match.distance_px if unit == "px" else match.distance_m) is not None
        )
        for reduction, value in (
            ("mean", math.fsum(v / len(values) for v in values) if values else None),
            ("median", float(np.median(values)) if values else None),
            ("p95", float(np.quantile(values, 0.95, method="linear")) if values else None),
        ):
            metrics += (
                replace(
                    _metric(
                        "matched_centroid_error_" + unit + "_" + reduction,
                        value,
                        len(values),
                        aggregation="matched_component_conditional_" + reduction,
                        unit="pixel" if unit == "px" else "m",
                        reason="no eligible matched components carrying requested geometry",
                    ),
                    support=MetricSupport(unit="COMPONENT", n=len(values)),
                ),
            )
    return metrics


def _component_bootstrap(
    observations: tuple[ComponentObservation, ...],
    cfg: GeneralizationConfig,
) -> BootstrapEvidence:
    """Resample parent groups jointly, preserving components and related variants together."""
    point = _component_metrics(observations)
    groups: dict[str, list[ComponentObservation]] = defaultdict(list)
    for observation in observations:
        groups[observation.row.group_id].append(observation)
    cohorts = tuple(tuple(cohort) for _, cohort in sorted(groups.items()))
    estimates: dict[str, list[float]] = {metric.name: [] for metric in point}
    attempted = 0
    if len(cohorts) >= 2:
        rng = np.random.default_rng(cfg.seed)
        for _ in range(cfg.bootstrap_replicates):
            selected = rng.integers(0, len(cohorts), size=len(cohorts))
            replicate = tuple(row for index in selected for row in cohorts[int(index)])
            for metric in _component_metrics(replicate):
                if metric.value is not None:
                    estimates[metric.name].append(metric.value)
            attempted += 1
    metrics: list[MetricValue] = []
    audits: list[IntervalAudit] = []
    for metric in point:
        values = estimates[metric.name]
        reason = (
            "fewer than two recorded groups"
            if len(cohorts) < 2
            else "point metric unavailable"
            if metric.value is None
            else "fewer than two valid support replicates"
            if len(values) < 2
            else None
        )
        interval = None
        if reason is None:
            lower, upper = np.quantile(
                values, ((1 - cfg.confidence) / 2, (1 + cfg.confidence) / 2), method="linear"
            )
            interval = ConfidenceInterval(
                lower=float(lower),
                upper=float(upper),
                confidence=cfg.confidence,
                method="percentile_recorded_group_bootstrap",
                n_replicates=cfg.bootstrap_replicates,
                n_valid=len(values),
                seed=cfg.seed,
            )
        metrics.append(replace(metric, interval=interval))
        audits.append(
            IntervalAudit(
                metric.name,
                cfg.bootstrap_replicates,
                attempted,
                len(values),
                attempted - len(values),
                len(cohorts),
                reason,
            )
        )
    return BootstrapEvidence(tuple(metrics), tuple(audits))


def _component_strata(
    rows: tuple[CaptureRow, ...],
    cfg: GeneralizationConfig,
) -> dict[tuple[str, str | None], tuple[ComponentObservation, ...]]:
    """Slice only unfiltered truth component size, including misses and explicit metre
    missingness."""
    cohorts: dict[tuple[str, str | None], list[ComponentObservation]] = defaultdict(list)
    if any(row.spatial is None for row in rows):
        return {}
    for row in rows:
        if row.spatial is None:
            continue
        for component in row.spatial.localization.truth:
            observation = ComponentObservation(row, component.component_id)
            size = _bin(component.area_px, cfg.size_edges_px)
            for key in (
                ("truth_component_area_px", size),
                (
                    "truth_component_area_m2",
                    _bin(component.area_m2, cfg.size_edges_m2)
                    if component.area_m2 is not None
                    else None,
                ),
                (
                    "gsd_lateral_by_truth_component_area_px",
                    f"{_bin(row.gsd_m[0], cfg.gsd_edges_m)}|{size}",
                ),
            ):
                cohorts[key].append(observation)
    return {key: tuple(cohort) for key, cohort in cohorts.items()}


def fit_baseline(rows: tuple[CaptureRow, ...]) -> Result[BaselineSpec, str]:
    """Fit only the frozen canonical train cohort; held-out labels cannot fit this policy."""
    checked = _validate(rows)
    if isinstance(checked, Err):
        return checked
    if rows[0].key.split != "train":
        return Err("baseline fitting requires canonical train rows, not validation or test")
    prevalence = (
        sum(row.label for row in rows) / len(rows) if rows[0].key.task == "classifier" else None
    )
    return Ok(
        BaselineSpec(
            rows[0].key.task,
            rows[0].key.dataset_hash,
            len(rows),
            len({r.group_id for r in rows}),
            prevalence,
        )
    )


def _bin(value: float, edges: tuple[float, ...]) -> str:
    """Use left-closed, right-open bins, including an explicit overflow cohort at the last edge."""
    if not edges:
        return "unsliced"
    if value < edges[0]:
        return f"<{edges[0]:g}"
    for low, high in zip(edges, edges[1:]):
        if low <= value < high:
            return f"[{low:g},{high:g})"
    return f">={edges[-1]:g}"


def _strata(
    rows: tuple[CaptureRow, ...],
    cfg: GeneralizationConfig,
) -> dict[tuple[str, str | None], tuple[CaptureRow, ...]]:
    """Partition exact captured metadata and truth-image area; never use predicted size."""
    names = sorted({tag.name for row in rows for tag in row.metadata.conditions})
    partitions: dict[tuple[str, str | None], list[CaptureRow]] = defaultdict(list)
    for row in rows:
        metadata = row.metadata
        tags = {tag.name: tag.value for tag in metadata.conditions}
        area = _values(row).get("target_area_px") if row.key.task == "segmentor" else None
        ground_area = _values(row).get("target_area_m2") if row.key.task == "segmentor" else None
        size = _bin(area, cfg.size_edges_px) if area is not None else None
        fields: tuple[tuple[str, str | None], ...] = (
            ("gsd_bin", row.bin_id or None),
            ("gsd_lateral_m", _bin(row.gsd_m[0], cfg.gsd_edges_m)),
            ("gsd_along_m", _bin(row.gsd_m[1], cfg.gsd_edges_m)),
            ("gsd_pair_m", f"{row.gsd_m[0]!r},{row.gsd_m[1]!r}"),
            ("gsd_anisotropy", repr(max(row.gsd_m) / min(row.gsd_m))),
            (
                "gsd_provenance",
                None
                if row.gsd_nominal is None
                else "nominal"
                if row.gsd_nominal
                else "recorded_non_nominal",
            ),
            (
                "acquisition_month_utc",
                metadata.acquired_at_utc[:7] if metadata.acquired_at_utc else None,
            ),
            ("truth_image_area_px", size),
            (
                "truth_image_area_m2",
                _bin(ground_area, cfg.size_edges_m2) if ground_area is not None else None,
            ),
            (
                "gsd_lateral_by_truth_image_area_px",
                f"{_bin(row.gsd_m[0], cfg.gsd_edges_m)}|{size}" if size is not None else None,
            ),
        ) + tuple(("condition:" + name, tags.get(name)) for name in names)
        for field in fields:
            partitions[field].append(row)
    return {key: tuple(values) for key, values in partitions.items()}


def _baseline_metrics(
    rows: tuple[CaptureRow, ...],
    score: ScoreConfig,
    baseline: BaselineSpec,
) -> Result[tuple[tuple[MetricValue, ...], tuple[CurveEvidence, ...]], str]:
    """Apply the fixed train policy without refitting held-out outcomes."""
    if (
        not is_sha256(baseline.training_dataset_hash)
        or not 1 <= baseline.n_groups <= baseline.n_images
    ):
        return Err("frozen baseline lacks valid train identity/support")
    if baseline.task != rows[0].key.task:
        return Err("baseline task differs from evaluation task")
    if baseline.task == "classifier":
        if baseline.training_prevalence is None:
            return Err("classifier baseline lacks frozen train prevalence")
        measured = score_classifier_baseline(
            baseline.training_prevalence,
            tuple(row.label for row in rows),
            replace(score, probability_thresholds=(score.classifier_probability_threshold,)),
        )
        if isinstance(measured, Err):
            return measured
        return Ok((measured.value.metrics, measured.value.curves))
    areas = tuple(_values(row).get("target_area_px") for row in rows)
    if any(area is None for area in areas):
        return Err("background baseline requires captured explicit mask area")
    positive = sum(area is not None and area > 0 for area in areas)
    return Ok(
        (
            (
                _metric(
                    "foreground_iou_mean_positive_images",
                    0.0 if positive else None,
                    positive,
                    aggregation="equal_image_mean",
                ),
                _metric(
                    "foreground_iou_mean_all_annotated_images",
                    sum(area == 0 for area in areas) / len(areas),
                    len(areas),
                    aggregation="equal_image_mean",
                ),
            ),
            (),
        )
    )


def measure_generalization(
    rows: tuple[CaptureRow, ...],
    score: ScoreConfig,
    cfg: GeneralizationConfig,
    *,
    baseline: BaselineSpec | None = None,
) -> Result[GeneralizationEvidence, str]:
    """Freeze exact recorded/truth-image strata and seeded grouped intervals once.

    No original date/condition, classifier plume size, temporal holdout or
    unseen-GSD extrapolation is inferred. Per-stratum pixel histograms are not
    reconstructed from compact scalars. Baselines must already be train-fitted.
    """
    measured = group_bootstrap(rows, score, cfg)
    if isinstance(measured, Err):
        return measured
    try:
        strata: list[StratumEvidence] = []
        audits: list[StratumIntervals] = []
        ordered = _ordered(rows)
        reduced_cohorts: dict[tuple[SampleKey, ...], BootstrapEvidence] = {
            tuple(row.key for row in ordered): measured.value,
        }
        partitions = _strata(ordered, cfg)
        for (name, value), cohort in sorted(
            partitions.items(),
            key=lambda item: (item[0][0], item[0][1] is not None, item[0][1] or ""),
        ):
            key = tuple(row.key for row in cohort)
            reduced = reduced_cohorts.get(key)
            if reduced is None:
                sampled = group_bootstrap(cohort, score, cfg)
                if isinstance(sampled, Err):
                    return sampled
                reduced = sampled.value
                reduced_cohorts[key] = reduced
            support = MetricSupport(
                unit="IMAGE",
                n=len(cohort),
                counts=(NamedCount(name="n_groups", value=len({r.group_id for r in cohort})),),
            )
            strata.append(StratumEvidence(name, value, reduced.metrics, support))
            audits.append(StratumIntervals(name, value, reduced.intervals))
        reduced_components: dict[tuple[tuple[SampleKey, int], ...], BootstrapEvidence] = {}
        for (name, value), components in sorted(
            _component_strata(ordered, cfg).items(),
            key=lambda item: (item[0][0], item[0][1] is not None, item[0][1] or ""),
        ):
            component_key = tuple(
                (component.row.key, component.truth_id) for component in components
            )
            component_reduction = reduced_components.get(component_key)
            if component_reduction is None:
                component_reduction = _component_bootstrap(components, cfg)
                reduced_components[component_key] = component_reduction
            support = MetricSupport(
                unit="COMPONENT",
                n=len(components),
                counts=(
                    NamedCount(name="n_groups", value=len({c.row.group_id for c in components})),
                ),
            )
            strata.append(StratumEvidence(name, value, component_reduction.metrics, support))
            audits.append(StratumIntervals(name, value, component_reduction.intervals))
        baseline_metrics: tuple[MetricValue, ...] = ()
        baseline_curves: tuple[CurveEvidence, ...] = ()
        if baseline is not None:
            scored = _baseline_metrics(rows, score, baseline)
            if isinstance(scored, Err):
                return scored
            baseline_metrics, baseline_curves = scored.value
        intervals_available = any(metric.interval is not None for metric in measured.value.metrics)
        interval_output = AvailabilityRecord(
            name="grouped_intervals",
            status="AVAILABLE" if intervals_available else "UNAVAILABLE",
            reason=None
            if intervals_available
            else "insufficient recorded-group or valid-replicate support; inspect interval audit",
        )
        outputs: tuple[AvailabilityRecord, ...] = (
            (
                interval_output,
                AvailabilityRecord(
                    name="classifier_truth_size",
                    status="UNAVAILABLE",
                    reason="classifier captures do not store explicit mask geometry",
                ),
            )
            if rows[0].key.task == "classifier"
            else (
                interval_output,
                AvailabilityRecord(
                    name="truth_component_sizes",
                    status="AVAILABLE"
                    if all(r.spatial is not None for r in rows)
                    and any(r.spatial is not None and r.spatial.localization.truth for r in rows)
                    else "UNAVAILABLE",
                    reason=None
                    if all(r.spatial is not None for r in rows)
                    and any(r.spatial is not None and r.spatial.localization.truth for r in rows)
                    else "complete component records and nonempty truth components are required",
                ),
                AvailabilityRecord(
                    name="stratum_pixel_ranking",
                    status="UNAVAILABLE",
                    reason=(
                        "compact scalar captures do not retain per-stratum pixel ranking histograms"
                    ),
                ),
            )
        )
        outputs += (
            AvailabilityRecord(
                name="training_baseline",
                status="AVAILABLE" if baseline is not None else "UNAVAILABLE",
                reason=None
                if baseline is not None
                else "no frozen train-derived baseline supplied",
            ),
            AvailabilityRecord(
                name="conditions",
                status="AVAILABLE" if any(r.metadata.conditions for r in rows) else "UNAVAILABLE",
                reason=None
                if any(r.metadata.conditions for r in rows)
                else "no categorical conditions recorded",
            ),
            AvailabilityRecord(
                name="timestamps",
                status="AVAILABLE"
                if any(r.metadata.acquired_at_utc for r in rows)
                else "UNAVAILABLE",
                reason=None
                if any(r.metadata.acquired_at_utc for r in rows)
                else "no acquisition timestamps recorded",
            ),
        )
        warnings = (
            "Recorded group IDs do not establish statistically independent observations.",
            "Whole groups are resampled; related source/GSD variants and components "
            "travel together.",
            "Grouped-random split coverage is not a temporal or unseen-GSD holdout.",
            "Intervals with invalid class/support replicates are conditional on the "
            "reported valid replicates.",
            "Metre sizes and errors use local-GSD approximations; non-nominal GSD is "
            "not proof of calibration.",
        )
        return Ok(
            GeneralizationEvidence(
                rows[0].key.dataset_hash,
                rows[0].dataset_manifest_hash,
                rows[0].key.task,
                rows[0].key.split,
                score,
                cfg,
                measured.value.metrics,
                measured.value.intervals,
                tuple(strata),
                tuple(audits),
                baseline,
                baseline_metrics,
                baseline_curves,
                outputs,
                warnings,
            )
        )
    except (ValueError, TypeError, OverflowError, RuntimeError) as exc:
        return Err(f"generalization measurement failed: {exc}")


def metric_reference(names: tuple[str, ...]) -> tuple[MetricReference, ...]:
    """Freeze explicit scientific metadata, leaving unknown definitions unavailable."""
    entries: list[MetricReference] = []
    for name in sorted(set(names)):
        definition = metric_definition(name)
        entries.append(
            MetricReference(
                name,
                definition.value if isinstance(definition, Ok) else None,
                definition.error if isinstance(definition, Err) else None,
            )
        )
    return tuple(entries)


def development_gaps(
    train: tuple[CaptureRow, ...],
    validation: tuple[CaptureRow, ...],
    score: ScoreConfig,
    *,
    checkpoint_hash: str,
) -> Result[DevelopmentEvidence, str]:
    """Compare exact scalar reductions for one caller-bound selected checkpoint, never test curves.

    The caller must establish that both captures describe ``checkpoint_hash``.
    Dataset/task/scoring populations must agree. Related groups cannot cross
    the train/validation boundary. Original metric supports and unavailable
    reasons survive alongside the explicit train-minus-validation difference.
    """
    if not is_sha256(checkpoint_hash):
        return Err("development gaps require a selected checkpoint SHA-256 identity")
    first, second = score_captured_rows(train, score), score_captured_rows(validation, score)
    if isinstance(first, Err):
        return first
    if isinstance(second, Err):
        return second
    if train[0].key.split != "train" or validation[0].key.split != "val":
        return Err("development gaps require train and validation, not test")
    if (train[0].key.task, train[0].key.dataset_hash, train[0].dataset_manifest_hash) != (
        validation[0].key.task,
        validation[0].key.dataset_hash,
        validation[0].dataset_manifest_hash,
    ):
        return Err("development gaps require the same task and dataset identity")
    if {row.group_id for row in train} & {row.group_id for row in validation}:
        return Err("development gaps refuse recorded-group leakage")
    observations_a = {row.metadata.observation_id for row in train} - {None}
    observations_b = {row.metadata.observation_id for row in validation} - {None}
    if observations_a & observations_b:
        return Err("development gaps refuse recorded-observation leakage")
    known = {metric.name: metric for metric in second.value}
    gaps: list[DevelopmentGap] = []
    for metric in first.value:
        other = known.get(metric.name)
        if other is None:
            continue
        compatible = (metric.unit, metric.aggregation, metric.threshold) == (
            other.unit,
            other.aggregation,
            other.threshold,
        )
        available = compatible and metric.value is not None and other.value is not None
        difference = (
            metric.value - other.value
            if metric.value is not None and other.value is not None and compatible
            else None
        )
        if difference is not None and not math.isfinite(difference):
            return Err("development gap exceeds finite numerical range")
        reason = (
            None
            if available
            else "metric definitions/thresholds differ"
            if not compatible
            else "one cohort metric is unavailable"
        )
        gaps.append(DevelopmentGap(metric.name, metric, other, difference, reason))
    return Ok(
        DevelopmentEvidence(
            train[0].key.dataset_hash,
            checkpoint_hash,
            tuple(gaps),
            train[0].dataset_manifest_hash,
            score,
            train[0].key.task,
        )
    )
