"""Dataset numerical measurements and the analysis orchestration boundary.

Explicit masks yield unfiltered four-connected component and local-GSD area
measurements. Pixel moments use every supplied pixel with population variance,
stable block merging, and exact equal-width unit-domain histogram counts.
Whole-dataset measurement freezes exact canonical tile/GSD variants into a
``DatasetMeasurement`` without sampling or content-based observation
inference: related task/augmentation copies collapse into one variant while
duplicate-content cohorts and missing annotations/provenance stay recorded,
never inferred. Summary assembly binds that frozen evidence to config and
code identity. Rendering and CLI activation remain separate phases.

Contains:
  - measure_mask: geometry of one explicit binary mask.
  - measure_pixels: per-band moments, endpoint counts, histograms and correlations.
  - measure_dataset: exact unsampled whole-dataset measurement.
  - dataset_summary: canonical summary assembly over frozen evidence.
  - analyze_dataset: the public ``Result`` measure-and-publish boundary.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import cast

import numpy as np
from flight.libs.types import Err, Ok, Result
from scipy import ndimage

from tools.ml_models.analysis.artifacts import dataset_identity
from tools.ml_models.analysis.config import DatasetAnalysisConfig, config_digest
from tools.ml_models.analysis.contracts import (
    ArtifactRef,
    AvailabilityRecord,
    CodeIdentity,
    DatasetIdentity,
    MetricSupport,
    MetricValue,
    NamedCount,
    SampleKey,
    Split,
    SplitEvidence,
    Task,
)
from tools.ml_models.analysis.summaries import DatasetSummary
from tools.ml_models.dataset.augment import ELEMENT_NAMES, apply_dihedral, legal_elements
from tools.ml_models.dataset.manifest import DatasetManifest, load_manifest, shard_dir
from tools.ml_models.dataset.store import RowRecord, read_rows


@dataclass(frozen=True, slots=True)
class MaskMeasurement:
    """Geometry of one explicit prepared mask; components are not physical plumes."""

    area_px: int
    area_fraction: float
    area_m2: float
    n_components: int
    component_areas_px: tuple[int, ...]
    border_touching: bool
    area_method: str = "local_lateral_gsd_times_along_gsd"
    connectivity: int = 4


def measure_mask(mask: np.ndarray, gsd_m: tuple[float, float]) -> Result[MaskMeasurement, str]:
    """Measure an explicit uint8 binary ``(1,H,W)`` mask without an area filter.

    Empty explicit masks have zero area/components; missing masks are invalid
    inputs, never manufactured negatives. Metre-squared area is the local
    approximation ``foreground_pixels * lateral_GSD * along_GSD``.
    """
    if (
        not isinstance(mask, np.ndarray)
        or mask.dtype != np.uint8
        or mask.ndim != 3
        or mask.shape[0] != 1
        or min(mask.shape[1:]) < 1
        or not bool(((mask == 0) | (mask == 1)).all())
    ):
        return Err("dataset mask measurement requires explicit binary uint8 (1,H,W)")
    try:
        if len(gsd_m) != 2 or any(
            isinstance(value, bool | np.bool_) or not math.isfinite(value) or value <= 0
            for value in gsd_m
        ):
            return Err("dataset mask area requires finite positive GSD on both axes")
        foreground = mask[0] != 0
        labelled, n = ndimage.label(foreground, structure=ndimage.generate_binary_structure(2, 1))
        areas = tuple(int(value) for value in np.bincount(labelled.ravel())[1:])
        area = int(np.count_nonzero(foreground))
        ground_area = area * gsd_m[0] * gsd_m[1]
        if not math.isfinite(ground_area):
            return Err("local GSD mask area exceeds finite floating-point range")
        border = bool(
            foreground[0, :].any()
            or foreground[-1, :].any()
            or foreground[:, 0].any()
            or foreground[:, -1].any()
        )
        return Ok(
            MaskMeasurement(
                area_px=area,
                area_fraction=area / foreground.size,
                area_m2=ground_area,
                n_components=int(n),
                component_areas_px=areas,
                border_touching=border,
            )
        )
    except (TypeError, ValueError, OverflowError, RuntimeError) as exc:
        return Err(f"dataset mask measurement failed: {exc}")


@dataclass(frozen=True, slots=True)
class BandMeasurement:
    """Population moments and exact processed-unit-pixel endpoint/histogram counts."""

    name: str
    mean: float
    std: float
    minimum: float
    maximum: float
    at_zero: int
    at_one: int
    histogram_counts: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PixelMeasurement:
    """Pixel-weighted measurements of the supplied image cohort, not independent samples."""

    n_images: int
    n_pixels: int
    bands: tuple[BandMeasurement, ...]
    histogram_bin_edges: tuple[float, ...]
    correlations: tuple[tuple[float | None, ...], ...]
    method: str = "exact_provided_pixel_traversal_population_moments"
    limitations: tuple[str, ...] = (
        "Pixels and related source variants are correlated, not independent observations.",
        "Moments/correlations are float64 calculations over every supplied pixel.",
        "Endpoint counts describe processed unit pixels, not raw sensor saturation.",
        "Constant-band correlations are unavailable (null), including the diagonal.",
        "Cohort selection belongs to the caller; this helper does not deduplicate augmentations.",
    )


def measure_pixels(
    images: Iterable[np.ndarray],
    band_names: tuple[str, ...],
    *,
    histogram_bins: int = 32,
) -> Result[PixelMeasurement, str]:
    """Measure finite unit float32 ``(C,H,W)`` images, allowing unequal spatial sizes.

    Each pixel has equal weight across images. Covariance and population
    variance merge centered image blocks in float64 rather than subtracting
    large raw squared moments. Histograms cover [0,1] with left-closed bins
    and a right-closed last bin. No distribution sampling or RNG is used.
    """
    if (
        not band_names
        or len(set(band_names)) != len(band_names)
        or any(not isinstance(name, str) or not name.strip() for name in band_names)
        or type(histogram_bins) is not int
        or histogram_bins < 2
    ):
        return Err("pixel measurement requires nonempty images, unique bands and at least two bins")
    channels = len(band_names)
    edges = np.linspace(0.0, 1.0, histogram_bins + 1, dtype=np.float64)
    n = 0
    n_images = 0
    mean = np.zeros(channels, dtype=np.float64)
    centered_products = np.zeros((channels, channels), dtype=np.float64)
    minima = np.full(channels, np.inf, dtype=np.float64)
    maxima = np.full(channels, -np.inf, dtype=np.float64)
    at_zero = [0] * channels
    at_one = [0] * channels
    counts = [[0] * histogram_bins for _ in band_names]
    try:
        for image in images:
            if (
                not isinstance(image, np.ndarray)
                or image.dtype != np.float32
                or image.ndim != 3
                or image.shape[0] != channels
                or min(image.shape[1:]) < 1
                or not bool(np.isfinite(image).all())
                or not bool(((image >= 0.0) & (image <= 1.0)).all())
            ):
                return Err("pixel measurement requires finite unit float32 (C,H,W) band layout")
            n_images += 1
            pixels = image.reshape(channels, -1).astype(np.float64)
            block_n = pixels.shape[1]
            block_mean = pixels.mean(axis=1)
            centered = pixels - block_mean[:, None]
            delta = block_mean - mean
            combined_n = n + block_n
            centered_products += centered @ centered.T + np.outer(delta, delta) * (
                n * (block_n / combined_n)
            )
            mean += delta * (block_n / combined_n)
            n = combined_n
            minima = np.minimum(minima, pixels.min(axis=1))
            maxima = np.maximum(maxima, pixels.max(axis=1))
            for index in range(channels):
                at_zero[index] += int(np.count_nonzero(pixels[index] == 0.0))
                at_one[index] += int(np.count_nonzero(pixels[index] == 1.0))
                histogram, _ = np.histogram(pixels[index], bins=edges)
                for bin_index, value in enumerate(histogram):
                    counts[index][bin_index] += int(value)
        if n_images == 0:
            return Err("pixel measurement requires nonempty images")
        diagonal = np.diag(centered_products)
        bands = tuple(
            BandMeasurement(
                name=name,
                mean=float(mean[index]),
                std=math.sqrt(float(diagonal[index]) / n),
                minimum=float(minima[index]),
                maximum=float(maxima[index]),
                at_zero=at_zero[index],
                at_one=at_one[index],
                histogram_counts=tuple(counts[index]),
            )
            for index, name in enumerate(band_names)
        )
        correlations = tuple(
            tuple(
                None
                if diagonal[first] == 0 or diagonal[second] == 0
                else max(
                    -1.0,
                    min(
                        1.0,
                        float(
                            centered_products[first, second]
                            / math.sqrt(diagonal[first] * diagonal[second])
                        ),
                    ),
                )
                for second in range(channels)
            )
            for first in range(channels)
        )
        return Ok(
            PixelMeasurement(
                n_images=n_images,
                n_pixels=n,
                bands=bands,
                histogram_bin_edges=tuple(float(value) for value in edges),
                correlations=correlations,
            )
        )
    except (TypeError, ValueError, OverflowError, RuntimeError, FloatingPointError) as exc:
        return Err(f"dataset pixel measurement failed: {exc}")


@dataclass(frozen=True, slots=True)
class DatasetSample:
    """One canonical tile/GSD variant, unioned across tasks and augmentation copies.

    ``key`` points to the preferred stored view, not a manufactured identity row.
    Images and masks are inverted to source orientation for measurements.
    ``mask`` is null without stored explicit geometry, even if a classifier
    sidecar records that its source had a prepared mask.
    """

    variant_id: str
    key: SampleKey
    row: RowRecord
    label: int
    gsd_m: tuple[float, float]
    tasks: tuple[Task, ...]
    stored_rows: int
    image_sha256: str
    mask: MaskMeasurement | None
    gsd_anisotropy: float
    tile_area_m2: float
    mask_key: SampleKey | None


@dataclass(frozen=True, slots=True)
class DatasetComponent:
    """One unfiltered four-connected explicit-mask component; local-GSD area."""

    variant_id: str
    component_index: int
    area_px: int
    area_m2: float


@dataclass(frozen=True, slots=True)
class CoverageCount:
    """Exact categorical support; missing categories are explicit, not inferred."""

    population: str
    split: Split | None
    field: str
    value: str | None
    n: int
    total: int


@dataclass(frozen=True, slots=True)
class PixelCohort:
    """Unsampled pixel-weighted evidence for canonical variants, unioned across tasks."""

    split: Split | None
    pixels: PixelMeasurement


@dataclass(frozen=True, slots=True)
class BaselineEvidence:
    """Held-out descriptors with fixed train-derived probability or all-background masks."""

    task: Task
    split: Split
    metrics: tuple[MetricValue, ...]


@dataclass(frozen=True, slots=True)
class ContentDuplicate:
    """Identical source-oriented image bytes across distinct canonical variants.

    Different GSDs or annotations can still share pixels. This is a diagnostic
    content identity, not proof of a shared observation or statistical independence.
    """

    image_sha256: str
    variant_ids: tuple[str, ...]
    splits: tuple[Split, ...]
    n_variants: int


@dataclass(frozen=True, slots=True)
class DatasetMeasurement:
    """Frozen measurement inputs for summary assembly and rendering without traversal."""

    identity: DatasetIdentity
    manifest: DatasetManifest
    samples: tuple[DatasetSample, ...]
    components: tuple[DatasetComponent, ...]
    metrics: tuple[MetricValue, ...]
    splits: tuple[SplitEvidence, ...]
    coverage: tuple[CoverageCount, ...]
    pixels: tuple[PixelCohort, ...]
    baselines: tuple[BaselineEvidence, ...]
    duplicates: tuple[ContentDuplicate, ...]
    warnings: tuple[str, ...]
    method: str = "exact_canonical_tile_gsd_variants_v1"


@dataclass(slots=True)
class _Variant:
    """Temporary exact row reconciliation; never retains full image arrays."""

    sample: DatasetSample
    elements: dict[Task, set[str]] = field(default_factory=dict)
    mask_hash: str | None = None


def _inverse(array: np.ndarray, element: str) -> np.ndarray:
    """Return a dtype-preserving source-oriented channel-first array."""
    inverse = "rot270" if element == "rot90" else "rot90" if element == "rot270" else element
    return apply_dihedral(array, inverse)


def _array_hash(array: np.ndarray) -> str:
    """Hash dtype, exact shape, and contiguous array bytes, without numeric coercion."""
    digest = hashlib.sha256(f"{array.dtype.str}:{array.shape}:".encode("ascii"))
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _metric(
    name: str,
    value: float | int | None,
    n: int,
    *,
    unit: str = "count",
    aggregation: str = "exact_canonical_variant_count",
    reason: str = "no eligible support",
) -> MetricValue:
    """Construct a finite scalar with its explicit population and undefined reason."""
    return MetricValue(
        name=name,
        value=value,
        status="UNAVAILABLE" if value is None else "AVAILABLE",
        reason=reason if value is None else None,
        unit=unit,
        aggregation=aggregation,
        support=MetricSupport(unit="IMAGE", n=n),
    )


def _mask_category(sample: DatasetSample) -> str:
    """Distinguish missing, unknown geometry, positive-empty and verified negative masks."""
    if sample.mask is None:
        return "missing" if sample.row.prepared_mask_state == "MISSING" else "geometry_unavailable"
    if sample.mask.area_px:
        return "positive_nonempty" if sample.label else "negative_nonempty"
    return "positive_empty" if sample.label else "verified_negative_empty"


def _cohort_counts(samples: tuple[DatasetSample, ...]) -> tuple[MetricValue, ...]:
    """Count variants once; original-observation totals require complete recorded identity."""
    n = len(samples)
    observations = {sample.row.metadata.observation_id for sample in samples} - {None}
    missing = sum(sample.row.metadata.observation_id is None for sample in samples)
    stored = sum(sample.stored_rows for sample in samples)
    tasks = sum(len(sample.tasks) for sample in samples)
    counts = (
        ("stored_rows", stored),
        ("canonical_task_variants", tasks),
        ("canonical_variants", n),
        ("augmentation_copies", stored - tasks),
        ("split_groups", len({sample.row.group_id for sample in samples})),
        ("recorded_observations", len(observations)),
        ("variants_missing_observation_id", missing),
        ("positive_variants", sum(sample.label for sample in samples)),
        ("explicit_mask_variants", sum(sample.mask is not None for sample in samples)),
        ("nominal_gsd_variants", sum(sample.row.gsd_nominal for sample in samples)),
    )
    return tuple(_metric(name, count, n) for name, count in counts) + (
        _metric(
            "total_observations",
            len(observations) if not missing else None,
            n,
            aggregation="unique_recorded_source_observation_id",
            reason="some variants lack authoritative observation IDs; total cannot be inferred",
        ),
    )


def _coverage(samples: tuple[DatasetSample, ...]) -> tuple[CoverageCount, ...]:
    """Freeze exact variant and recorded-observation coverage, including absent conditions."""
    records: list[CoverageCount] = []
    condition_names = sorted({tag.name for s in samples for tag in s.row.metadata.conditions})
    for split in (None, "train", "val", "test"):
        selected = tuple(s for s in samples if split is None or s.key.split == split)
        observations: dict[str, DatasetSample] = {}
        for sample in selected:
            if sample.row.metadata.observation_id is not None:
                observations.setdefault(sample.row.metadata.observation_id, sample)
        for population, cohort in (
            ("canonical_variant", selected),
            ("recorded_observation", tuple(observations.values())),
        ):
            fields: dict[str, Counter[str | None]] = {
                name: Counter()
                for name in (
                    "timestamp",
                    "month_utc",
                    "source_annotation_state",
                    "annotation_source",
                    "observation_id",
                    *("condition:" + name for name in condition_names),
                )
            }
            if population == "canonical_variant":
                fields["mask_category"] = Counter()
                fields["prepared_mask_state"] = Counter()
                fields["gsd_provenance"] = Counter()
                fields["shape"] = Counter()
                fields["bin_id"] = Counter()
                fields["label"] = Counter()
            for sample in cohort:
                metadata = sample.row.metadata
                fields["timestamp"]["recorded" if metadata.acquired_at_utc else "missing"] += 1
                fields["month_utc"][
                    metadata.acquired_at_utc[:7] if metadata.acquired_at_utc else None
                ] += 1
                fields["source_annotation_state"][metadata.source_annotation_state] += 1
                fields["annotation_source"][
                    "recorded" if metadata.annotation_source else "missing"
                ] += 1
                fields["observation_id"]["recorded" if metadata.observation_id else "missing"] += 1
                tags = {tag.name: tag.value for tag in metadata.conditions}
                for name in condition_names:
                    fields["condition:" + name][tags.get(name)] += 1
                if population == "canonical_variant":
                    fields["mask_category"][_mask_category(sample)] += 1
                    fields["prepared_mask_state"][sample.row.prepared_mask_state] += 1
                    fields["gsd_provenance"][
                        "nominal" if sample.row.gsd_nominal else "recorded_non_nominal"
                    ] += 1
                    height, width = sample.key.spatial_shard
                    fields["shape"][f"{height}x{width}"] += 1
                    fields["bin_id"][sample.row.bin_id or None] += 1
                    fields["label"]["positive" if sample.label else "negative"] += 1
            records.extend(
                CoverageCount(population, split, name, value, count, len(cohort))
                for name, counts in fields.items()
                for value, count in sorted(
                    counts.items(), key=lambda item: (item[0] is not None, item[0] or "")
                )
            )
    return tuple(records)


def _baselines(samples: tuple[DatasetSample, ...]) -> tuple[BaselineEvidence, ...]:
    """Measure fixed canonical-train prevalence and explicit all-background descriptors."""
    train = tuple(s for s in samples if s.key.split == "train" and "classifier" in s.tasks)
    prevalence = sum(s.label for s in train) / len(train) if train else None
    records: list[BaselineEvidence] = []
    for task in ("classifier", "segmentor"):
        for split in ("train", "val", "test"):
            cohort = tuple(s for s in samples if s.key.split == split and task in s.tasks)
            if not cohort:
                continue
            n = len(cohort)
            if task == "classifier":
                losses = [
                    -math.log(prevalence)
                    if s.label and prevalence and prevalence > 0
                    else -math.log1p(-prevalence)
                    if not s.label and prevalence is not None and prevalence < 1
                    else None
                    for s in cohort
                ]
                bce = (
                    math.fsum(cast(float, loss) / n for loss in losses)
                    if all(loss is not None for loss in losses)
                    else None
                )
                metrics = (
                    _metric(
                        "baseline_train_prevalence",
                        prevalence,
                        len(train),
                        unit="dimensionless",
                        aggregation="canonical_classifier_train_positive_fraction",
                        reason="no classifier train cohort",
                    ),
                ) + tuple(
                    _metric(
                        name,
                        value,
                        n,
                        unit="dimensionless",
                        aggregation="equal_canonical_image_mean",
                        reason=reason,
                    )
                    for name, value, reason in (
                        (
                            "baseline_train_prevalence_brier",
                            math.fsum((prevalence - s.label) ** 2 / n for s in cohort)
                            if prevalence is not None
                            else None,
                            "no classifier train cohort",
                        ),
                        (
                            "baseline_train_prevalence_bce",
                            bce,
                            "train prevalence absent or assigns zero probability to a"
                            " held-out outcome; BCE is nonfinite",
                        ),
                        (
                            "baseline_always_negative_accuracy",
                            sum(not s.label for s in cohort) / n,
                            "",
                        ),
                        ("baseline_always_positive_accuracy", sum(s.label for s in cohort) / n, ""),
                        (
                            "baseline_train_majority_accuracy",
                            sum(s.label == int(prevalence >= 0.5) for s in cohort) / n
                            if prevalence is not None
                            else None,
                            "no classifier train cohort",
                        ),
                    )
                )
            else:
                masks = tuple(s.mask for s in cohort if s.mask is not None)
                positive = sum(mask.area_px > 0 for mask in masks)
                metrics = (
                    _metric(
                        "baseline_background_positive_iou",
                        0.0 if positive else None,
                        positive,
                        unit="dimensionless",
                        aggregation="equal_nonempty_truth_image_mean",
                    ),
                    _metric(
                        "baseline_background_all_annotated_iou",
                        sum(mask.area_px == 0 for mask in masks) / n,
                        n,
                        unit="dimensionless",
                        aggregation="equal_annotated_image_mean",
                    ),
                    replace(
                        _metric(
                            "baseline_background_false_negative_pixels",
                            sum(mask.area_px for mask in masks),
                            n,
                            unit="pixel",
                            aggregation="pooled_pixel_count",
                        ),
                        support=MetricSupport(
                            unit="PIXEL",
                            n=sum(s.key.spatial_shard[0] * s.key.spatial_shard[1] for s in cohort),
                            counts=(NamedCount(name="n_images", value=n),),
                        ),
                    ),
                )
            records.append(BaselineEvidence(task, split, metrics))
    return tuple(records)


def _canonical_images(root: Path, samples: tuple[DatasetSample, ...]) -> Iterator[np.ndarray]:
    """Stream canonical images using one read-only shard mapping at a time."""
    directory: Path | None = None
    images: np.ndarray | None = None
    for sample in samples:
        target = shard_dir(root, sample.key.task, sample.key.split, *sample.key.spatial_shard)
        if directory != target:
            images = np.load(target / "images.npy", mmap_mode="r", allow_pickle=False)
            directory = target
        if images is None:
            raise ValueError("canonical image mapping unavailable")
        yield _inverse(images[sample.key.row_index], sample.key.element)


def _scan_variants(
    root: Path,
    manifest: DatasetManifest,
    identity: DatasetIdentity,
) -> tuple[DatasetSample, ...]:
    """Validate every stored row and reconcile exact source-oriented task/augmentation copies."""
    variants: dict[str, _Variant] = {}
    groups: dict[str, str] = {}
    frames: dict[str, str] = {}
    tile_sources: dict[str, tuple[object, ...]] = {}
    observation_sources: dict[str, tuple[object, ...]] = {}
    declared: set[Path] = set()
    gsd_min = np.full(2, np.inf)
    gsd_max = np.zeros(2)
    for count in sorted(manifest.shards, key=lambda c: (c.task, c.split, c.height, c.width)):
        if count.task not in ("classifier", "segmentor") or count.split not in (
            "train",
            "val",
            "test",
        ):
            raise ValueError("dataset contains unknown task or split")
        task, split = cast(Task, count.task), cast(Split, count.split)
        directory = shard_dir(root, task, split, count.height, count.width)
        if directory in declared:
            raise ValueError("manifest repeats a spatial shard")
        declared.add(directory)
        rows = read_rows(directory)
        arrays = {
            name: np.load(directory / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            for name in ("images", "gsd", "labels")
        }
        if len(rows) != count.n:
            raise ValueError("dataset row count disagrees with manifest")
        for name, dtype, shape in (
            ("images", np.float32, (count.n, len(manifest.band_names), count.height, count.width)),
            ("gsd", np.float32, (count.n, 2)),
            ("labels", np.float32, (count.n, 1)),
        ):
            if arrays[name].dtype != dtype or arrays[name].shape != shape:
                raise ValueError(f"dataset {name} dtype/shape disagrees with manifest")
        if not np.isin(arrays["labels"], (0.0, 1.0)).all():
            raise ValueError("dataset labels must be exactly binary")
        if int(np.count_nonzero(arrays["labels"])) != count.n_positive:
            raise ValueError("dataset positive count disagrees with manifest")
        if not np.isfinite(arrays["gsd"]).all() or np.any(arrays["gsd"] <= 0):
            raise ValueError("dataset GSD must be finite positive metres")
        gsd_min = np.minimum(gsd_min, arrays["gsd"].min(axis=0))
        gsd_max = np.maximum(gsd_max, arrays["gsd"].max(axis=0))
        masks = (
            np.load(directory / "masks.npy", mmap_mode="r", allow_pickle=False)
            if task == "segmentor"
            else None
        )
        if masks is not None and (
            masks.dtype != np.uint8 or masks.shape != (count.n, 1, count.height, count.width)
        ):
            raise ValueError("explicit mask dtype/shape disagrees with manifest")
        if task == "classifier" and (directory / "masks.npy").exists():
            raise ValueError("classifier shards cannot store segmentation masks")
        for index, row in enumerate(rows):
            if not row.tile_id.strip() or not row.group_id.strip():
                raise ValueError("dataset tile and group identities must be nonblank")
            if row.element not in legal_elements(count.height, count.width):
                raise ValueError("dataset contains an illegal augmentation element")
            if split != "train" and row.element != "id":
                raise ValueError("held-out rows must be identity views")
            if split == "train" and row.element not in manifest.augment.elements:
                raise ValueError("train augmentation disagrees with manifest recipe")
            prior_split = groups.setdefault(row.group_id, split)
            if prior_split != split:
                raise ValueError("cross-task split group leakage")
            if row.frame_id is not None and frames.setdefault(row.frame_id, split) != split:
                raise ValueError("recorded source frame leakage across splits")
            label = int(arrays["labels"][index, 0])
            source = (split, row.group_id, label, row.metadata, row.frame_id, row.grid_rc)
            gsd = (float(arrays["gsd"][index, 0]), float(arrays["gsd"][index, 1]))
            tile_source = source + (
                row.bin_id,
                gsd,
                count.height,
                count.width,
                row.theta_g_deg,
                row.gsd_nominal,
                row.prepared_mask_state,
            )
            if tile_sources.setdefault(row.tile_id, tile_source) != tile_source:
                raise ValueError(
                    "tile identity has conflicting source metadata, labels, geometry or split"
                )
            observation = row.metadata.observation_id
            if (
                observation is not None
                and observation_sources.setdefault(observation, source) != source
            ):
                raise ValueError(
                    "recorded observation has conflicting source metadata, labels or split"
                )
            variant_id = hashlib.sha256(
                json.dumps(
                    (row.tile_id, row.bin_id, gsd, count.height, count.width),
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            image = _inverse(arrays["images"][index], row.element)
            if not np.isfinite(image).all() or np.any(image < 0) or np.any(image > 1):
                raise ValueError("dataset images require finite processed unit pixels")
            image_hash = _array_hash(image)
            mask_hash = None
            measurement = None
            if masks is not None:
                mask = _inverse(masks[index], row.element)
                measured = measure_mask(mask, gsd)
                if isinstance(measured, Err):
                    raise ValueError(measured.error)
                measurement = measured.value
                mask_hash = _array_hash(mask)
                state = "NONEMPTY" if measurement.area_px else "EMPTY"
                if row.prepared_mask_state not in ("UNKNOWN", state):
                    raise ValueError("prepared mask state disagrees with explicit geometry")
            if row.metadata.source_annotation_state in ("MISSING", "EXPLICIT_EMPTY") and (
                measurement is not None
                and measurement.area_px > 0
                or row.prepared_mask_state == "NONEMPTY"
                or row.metadata.source_annotation_state == "MISSING"
                and (row.prepared_mask_state == "EMPTY" or measurement is not None)
            ):
                raise ValueError("source annotation state contradicts prepared mask state")
            key = SampleKey(
                dataset_hash=identity.content_hash,
                task=task,
                split=split,
                spatial_shard=(count.height, count.width),
                row_index=index,
                tile_id=row.tile_id,
                element=row.element,
            )
            sample = DatasetSample(
                variant_id,
                key,
                row,
                label,
                gsd,
                (task,),
                1,
                image_hash,
                measurement,
                max(gsd) / min(gsd),
                count.height * count.width * gsd[0] * gsd[1],
                key if measurement is not None else None,
            )
            variant = variants.get(variant_id)
            if variant is None:
                variants[variant_id] = _Variant(sample, {task: {row.element}}, mask_hash)
                continue
            if row.element in variant.elements.setdefault(task, set()):
                raise ValueError("dataset repeats a task/tile/GSD augmentation identity")
            variant.elements[task].add(row.element)
            prior = variant.sample
            if (
                replace(prior.row, element="id") != replace(row, element="id")
                or prior.image_sha256 != image_hash
            ):
                raise ValueError(
                    "task/augmentation copies have conflicting metadata or canonical pixels"
                )
            if (
                variant.mask_hash is not None
                and mask_hash is not None
                and variant.mask_hash != mask_hash
            ):
                raise ValueError("augmentation copies have conflicting canonical masks")
            rank = (ELEMENT_NAMES.index(row.element), task)
            prior_rank = (ELEMENT_NAMES.index(prior.row.element), prior.key.task)
            preferred = sample if rank < prior_rank else prior
            variant.sample = replace(
                preferred,
                tasks=tuple(sorted(variant.elements)),
                stored_rows=prior.stored_rows + 1,
                mask=prior.mask if prior.mask is not None else measurement,
                mask_key=prior.mask_key if prior.mask is not None else sample.mask_key,
            )
            if mask_hash is not None:
                variant.mask_hash = mask_hash
    observed = {path.parent for path in root.rglob("images.npy")}
    if observed != declared:
        raise ValueError("dataset contains unlisted or missing image shards")
    if any(
        path.parent not in declared
        for name in ("gsd.npy", "labels.npy", "masks.npy", "rows.jsonl")
        for path in root.rglob(name)
    ):
        raise ValueError("dataset contains unlisted shard evidence")
    for variant in variants.values():
        sample = variant.sample
        expected = (
            set(manifest.augment.elements).intersection(legal_elements(*sample.key.spatial_shard))
            if sample.key.split == "train"
            else {"id"}
        )
        if any(elements != expected for elements in variant.elements.values()):
            raise ValueError("stored augmentation coverage disagrees with manifest recipe")
    for actual, recorded in (
        (gsd_min, (manifest.gsd_lateral_min_m, manifest.gsd_along_min_m)),
        (gsd_max, (manifest.gsd_lateral_max_m, manifest.gsd_along_max_m)),
    ):
        if not np.allclose(actual, recorded, rtol=1e-6, atol=0):
            raise ValueError("manifest GSD range disagrees with stored metre values")
    return tuple(variants[key].sample for key in sorted(variants))


def measure_dataset(root: Path) -> Result[DatasetMeasurement, str]:
    """Validate and freeze exact unsampled finished-dataset evidence without publishing.

    Every stored row is validated. Each tile/bin/GSD/shape is measured once
    across task/augmentation copies. Observation totals depend only on recorded
    IDs, never on parsing tile names. Duplicate-content cohorts are retained,
    not silently removed as if their original observations were known.
    """
    try:
        root = Path(root).resolve()
        if any(path.is_symlink() or os.path.isjunction(path) for path in root.rglob("*")):
            return Err("dataset analysis refuses linked dataset files/directories")
        verified = dataset_identity(root)
        if isinstance(verified, Err):
            return verified
        identity = verified.value
        manifest = load_manifest(root / "dataset.json", verify=False)
        samples = _scan_variants(root, manifest, identity)
        pixels: list[PixelCohort] = []
        for split in (None, "train", "val", "test"):
            cohort = tuple(s for s in samples if split is None or s.key.split == split)
            if not cohort:
                continue
            measured = measure_pixels(_canonical_images(root, cohort), identity.band_names)
            if isinstance(measured, Err):
                return measured
            pixels.append(PixelCohort(split, measured.value))
        components = tuple(
            DatasetComponent(s.variant_id, index, area, area * s.gsd_m[0] * s.gsd_m[1])
            for s in samples
            if s.mask is not None
            for index, area in enumerate(s.mask.component_areas_px)
        )
        contents: dict[str, list[DatasetSample]] = {}
        for sample in samples:
            contents.setdefault(sample.image_sha256, []).append(sample)
        duplicates = tuple(
            ContentDuplicate(
                digest,
                tuple(s.variant_id for s in cohort),
                tuple(sorted({s.key.split for s in cohort})),
                len(cohort),
            )
            for digest, cohort in sorted(contents.items())
            if len(cohort) > 1
        )
        baselines = _baselines(samples)
        splits: list[SplitEvidence] = []
        for task in ("classifier", "segmentor"):
            for split in ("train", "val", "test"):
                cohort = tuple(s for s in samples if task in s.tasks and s.key.split == split)
                if not cohort:
                    continue
                n_stored = sum(
                    count.n
                    for count in manifest.shards
                    if count.task == task and count.split == split
                )
                metrics = tuple(
                    metric
                    for metric in _cohort_counts(cohort)
                    if metric.name
                    not in ("stored_rows", "canonical_task_variants", "augmentation_copies")
                ) + (
                    _metric("stored_rows", n_stored, len(cohort)),
                    _metric("augmentation_copies", n_stored - len(cohort), len(cohort)),
                )
                baseline = next(b for b in baselines if b.task == task and b.split == split)
                splits.append(
                    SplitEvidence(
                        task=task,
                        split=cast(Split, split),
                        dataset_hash=identity.content_hash,
                        dataset_manifest_hash=identity.manifest_hash,
                        metrics=metrics + baseline.metrics,
                        support=MetricSupport(
                            unit="IMAGE",
                            n=len(cohort),
                            counts=(
                                NamedCount(name="stored_rows", value=n_stored),
                                NamedCount(
                                    name="n_groups", value=len({s.row.group_id for s in cohort})
                                ),
                            ),
                        ),
                    )
                )
        final_identity = dataset_identity(root)
        if isinstance(final_identity, Err):
            return final_identity
        if final_identity.value != identity:
            return Err("dataset changed during measurement")
        warnings = (
            "Recorded observations, related GSD variants, mask components and pixels"
            " do not establish statistical independence.",
            "Coverage describes stored source observations, not unavailable upstream"
            " source observations.",
            "GSD areas are local lateral-times-along approximations; non-nominal does"
            " not establish sensor calibration.",
            "Pixel statistics use all canonical variants without sampling; processed"
            " endpoints are not raw sensor saturation.",
            "Grouped-random split coverage is not evidence of temporal or unseen-GSD"
            " extrapolation.",
        ) + (
            (
                "Identical canonical image bytes occur across splits; inspect duplicate"
                " evidence before generalization claims.",
            )
            if any(len(d.splits) > 1 for d in duplicates)
            else ()
        )
        metrics = _cohort_counts(samples) + (
            _metric("duplicate_content_cohorts", len(duplicates), len(samples)),
            _metric(
                "duplicate_content_variants", sum(d.n_variants for d in duplicates), len(samples)
            ),
            _metric(
                "cross_split_duplicate_cohorts",
                sum(len(d.splits) > 1 for d in duplicates),
                len(samples),
            ),
            _metric("group_leakage", 0, len(samples)),
        )
        geometry = (
            ("gsd_lateral_m", tuple(s.gsd_m[0] for s in samples), "m"),
            ("gsd_along_m", tuple(s.gsd_m[1] for s in samples), "m"),
            ("gsd_anisotropy", tuple(s.gsd_anisotropy for s in samples), "dimensionless"),
            ("tile_area_m2", tuple(s.tile_area_m2 for s in samples), "m2"),
            (
                "positive_mask_area_px",
                tuple(
                    float(s.mask.area_px)
                    for s in samples
                    if s.mask is not None and s.mask.area_px > 0
                ),
                "pixel",
            ),
            (
                "positive_mask_area_m2",
                tuple(s.mask.area_m2 for s in samples if s.mask is not None and s.mask.area_px > 0),
                "m2",
            ),
            (
                "positive_mask_area_fraction",
                tuple(
                    s.mask.area_fraction
                    for s in samples
                    if s.mask is not None and s.mask.area_px > 0
                ),
                "dimensionless",
            ),
            (
                "explicit_mask_components",
                tuple(float(s.mask.n_components) for s in samples if s.mask is not None),
                "component",
            ),
            (
                "explicit_mask_border_touching",
                tuple(float(s.mask.border_touching) for s in samples if s.mask is not None),
                "dimensionless",
            ),
        )
        metrics += tuple(
            _metric(
                name + "_" + reduction,
                (
                    min(values)
                    if reduction == "min"
                    else max(values)
                    if reduction == "max"
                    else math.fsum(v / len(values) for v in values)
                )
                if values
                else None,
                len(values),
                unit=unit,
                aggregation="equal_canonical_variant_" + reduction,
            )
            for name, values, unit in geometry
            for reduction in ("min", "max", "mean")
        )
        return Ok(
            DatasetMeasurement(
                identity,
                manifest,
                samples,
                components,
                metrics,
                tuple(splits),
                _coverage(samples),
                tuple(pixels),
                baselines,
                duplicates,
                warnings,
            )
        )
    except (OSError, ValueError, TypeError, OverflowError, RuntimeError) as exc:
        return Err(f"dataset measurement failed: {exc}")


def dataset_summary(
    measured: DatasetMeasurement,
    cfg: DatasetAnalysisConfig,
    code: CodeIdentity,
    artifacts: tuple[ArtifactRef, ...],
) -> DatasetSummary:
    """Assemble the canonical summary from frozen values; never re-read dataset inputs.

    The identity binds the complete exact measurement, scientific settings,
    method version and recorded code identity. Optional provenance/figures can
    be unavailable without fabricating values or failing required numerical evidence.
    """
    digest = config_digest(cfg)
    measurement_id = hashlib.sha256(
        json.dumps(
            {"measurement": asdict(measured), "config_digest": digest, "code": asdict(code)},
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    timestamp = any(s.row.metadata.acquired_at_utc is not None for s in measured.samples)
    conditions = any(s.row.metadata.conditions for s in measured.samples)
    outputs = (
        AvailabilityRecord(name="dataset_measurements", status="AVAILABLE", required=True),
        AvailabilityRecord(name="tables", status="AVAILABLE", required=True),
        AvailabilityRecord(
            name="figures",
            status="UNAVAILABLE",
            reason="dataset rendering is a separate implementation phase",
        ),
        AvailabilityRecord(
            name="visuals",
            status="UNAVAILABLE",
            reason="dataset visual rendering is a separate implementation phase",
        ),
        AvailabilityRecord(
            name="timestamps",
            status="AVAILABLE" if timestamp else "UNAVAILABLE",
            reason=None if timestamp else "no acquisition timestamps were recorded",
        ),
        AvailabilityRecord(
            name="conditions",
            status="AVAILABLE" if conditions else "UNAVAILABLE",
            reason=None if conditions else "no categorical conditions were recorded",
        ),
    )
    warnings = measured.warnings + (
        ("Code is dirty or unknown; this bundle alone cannot reconstruct the source tree.",)
        if code.dirty is not False
        else ()
    )
    return DatasetSummary(
        measurement_id=measurement_id,
        dataset=measured.identity,
        code=code,
        config_digest=digest,
        metrics=measured.metrics,
        splits=measured.splits,
        artifacts=artifacts,
        outputs=outputs,
        warnings=warnings,
    )


def analyze_dataset(cfg: DatasetAnalysisConfig) -> Result[Path, str]:
    """Measure and publish one exclusive checksummed dataset evidence bundle.

    Invalid evidence, existing outputs and outputs inside the source are refused.
    Dataset rendering/CLI activation is a separate phase; optional figure
    availability is explicit in the measurement-only summary.
    """
    from tools.ml_models.analysis.dataset_artifacts import publish_dataset_measurement

    try:
        root, out = Path(cfg.dataset), Path(cfg.out)
        source, target = root.resolve(), out.resolve()
        if target == source or source in target.parents:
            return Err("dataset analysis output lies inside the source dataset")
        if out.exists():
            return Err("dataset analysis output already exists; refusing to overwrite")
    except (OSError, ValueError, RuntimeError) as exc:
        return Err(f"dataset analysis preflight failed: {exc}")
    measured = measure_dataset(root)
    if isinstance(measured, Err):
        return measured
    return publish_dataset_measurement(measured.value, cfg)
