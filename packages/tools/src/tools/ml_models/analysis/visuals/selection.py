"""Deterministic bounded dataset gallery selections over frozen evidence.

Seeded stratified round-robin representatives are display selections, not
random independent samples. GSD comparisons require recorded observation IDs;
no identity is inferred from a filename. Shared inputs are captured once.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass

from tools.ml_models.analysis.config import CaptureConfig
from tools.ml_models.analysis.contracts import AvailabilityRecord
from tools.ml_models.analysis.dataset import DatasetMeasurement, DatasetSample, _mask_category
from tools.ml_models.dataset.augment import legal_elements


@dataclass(frozen=True, slots=True)
class DatasetGallery:
    """One selected gallery; elements are verified stored training transforms."""

    identifier: str
    family: str
    variant_ids: tuple[str, ...]
    elements: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GalleryPlan:
    """Shared selected inputs, conservative byte reservation, and explicit family availability."""

    galleries: tuple[DatasetGallery, ...]
    variant_ids: tuple[str, ...]
    reserved_bytes: int
    outputs: tuple[AvailabilityRecord, ...]
    method: str = "seeded_stratified_round_robin_display_selection_v1"


def _rank(identity: str, seed: int) -> str:
    """Return a stable seeded display priority, without source identity inference."""
    return hashlib.sha256(f"{seed}:{identity}".encode()).hexdigest()


def _preview_reservation(sample: DatasetSample, channels: int) -> int:
    """Reserve raw float32 image/uint8 mask bytes plus a conservative NPZ header allowance."""
    height, width = sample.key.spatial_shard
    return height * width * (4 * channels + int(sample.mask_key is not None)) + 1024


def select_dataset_galleries(measured: DatasetMeasurement, cfg: CaptureConfig) -> GalleryPlan:
    """Select bounded representatives, stored transforms, and recorded same-observation GSD pairs.

    All decisions use frozen metadata only. Whole gallery groups must fit;
    pairs are never reduced to a misleading single GSD view. Inputs reused by
    multiple galleries consume the shared image/byte budget only once.
    """
    samples = sorted(measured.samples, key=lambda s: (_rank(s.variant_id, cfg.seed), s.variant_id))
    by_id = {s.variant_id: s for s in samples}
    selected: set[str] = set()
    reserved = 0
    galleries: list[DatasetGallery] = []

    def admit(ids: tuple[str, ...]) -> bool:
        """Atomically reserve new shared inputs for one complete display gallery."""
        nonlocal reserved
        extra = set(ids) - selected
        size = sum(
            _preview_reservation(by_id[key], len(measured.identity.band_names)) for key in extra
        )
        if (
            len(selected) + len(extra) > cfg.max_preview_images
            or reserved + size > cfg.max_capture_bytes
        ):
            return False
        selected.update(extra)
        reserved += size
        return True

    buckets: dict[tuple[str, int, str], list[DatasetSample]] = defaultdict(list)
    for sample in samples:
        buckets[(sample.key.split, sample.label, _mask_category(sample))].append(sample)
    representatives: list[DatasetSample] = []
    depth = 0
    while len(representatives) < cfg.examples_per_family:
        layer = [cohort[depth] for _, cohort in sorted(buckets.items()) if depth < len(cohort)]
        if not layer:
            break
        representatives.extend(layer[: cfg.examples_per_family - len(representatives)])
        depth += 1
    for sample in representatives:
        if admit((sample.variant_id,)):
            galleries.append(
                DatasetGallery(
                    "representative_" + sample.variant_id,
                    "representative",
                    (sample.variant_id,),
                )
            )
    training = [s for s in samples if s.key.split == "train"]
    for sample in training[: cfg.examples_per_family]:
        elements = tuple(
            element
            for element in measured.manifest.augment.elements
            if element in legal_elements(*sample.key.spatial_shard)
        )
        if len(elements) > 1 and admit((sample.variant_id,)):
            galleries.append(
                DatasetGallery(
                    "augmentation_" + sample.variant_id,
                    "augmentation",
                    (sample.variant_id,),
                    elements,
                )
            )
    observations: dict[str, list[DatasetSample]] = defaultdict(list)
    for sample in samples:
        if sample.row.metadata.observation_id is not None:
            observations[sample.row.metadata.observation_id].append(sample)
    pairs: list[tuple[str, tuple[str, str]]] = []
    for identity, variants in sorted(
        observations.items(), key=lambda item: _rank(item[0], cfg.seed)
    ):
        ordered = sorted(variants, key=lambda s: (s.gsd_m, s.variant_id))
        if ordered[0].gsd_m != ordered[-1].gsd_m:
            pairs.append((identity, (ordered[0].variant_id, ordered[-1].variant_id)))
    for identity, ids in pairs[: cfg.examples_per_family]:
        if admit(ids):
            galleries.append(
                DatasetGallery(
                    "gsd_" + hashlib.sha256(identity.encode("utf-8")).hexdigest(),
                    "same_observation_gsd",
                    ids,
                )
            )
    eligible = {
        "representative": len(samples),
        "augmentation": sum(
            len(
                set(measured.manifest.augment.elements).intersection(
                    legal_elements(*s.key.spatial_shard)
                )
            )
            > 1
            for s in training
        ),
        "same_observation_gsd": len(pairs),
    }
    outputs = tuple(
        AvailabilityRecord(
            name="gallery:" + family,
            status="AVAILABLE"
            if any(g.family == family for g in galleries)
            else "SKIPPED"
            if eligible[family]
            else "UNAVAILABLE",
            reason=None
            if any(g.family == family for g in galleries)
            else "No eligible recorded inputs for this gallery family"
            if not eligible[family]
            else "Gallery capture disabled or no complete gallery fit the configured shared budget",
        )
        for family in ("representative", "augmentation", "same_observation_gsd")
    )
    return GalleryPlan(tuple(galleries), tuple(sorted(selected)), reserved, outputs)
