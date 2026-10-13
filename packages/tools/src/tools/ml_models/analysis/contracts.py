"""Typed evidence records shared by evaluation, capture, and rendering.

These frozen records are the alignment keys and metric vocabulary for the
evidence-first analysis stack. Full schemas (curve, support, and result
records; validation) land with the evidence contract phase; only the minimal
typed scaffold contracts are declared here.

Contains:
  - SampleKey: dataset/split/shard/row identity for one scored sample.
  - MetricValue: one named measurement or an explicit unavailable state.
  - SplitEvidence: task/split/dataset identity plus aggregate metrics.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True, slots=True)
class SampleKey:
    """Alignment key joining scores, metadata, and previews for one row.

    Attributes:
        dataset_hash: Content hash of the source dataset.
        task: ``classifier`` or ``segmentor``.
        split: ``train``, ``val``, or ``test``.
        spatial_shard: ``(height, width)`` tile-shard identity.
        row_index: Row position within the shard.
        tile_id: Source tile identifier.
        element: Augmentation element within the row.
    """

    dataset_hash: str
    task: Literal["classifier", "segmentor"]
    split: Literal["train", "val", "test"]
    spatial_shard: tuple[int, int]
    row_index: int
    tile_id: str
    element: str


@dataclass(frozen=True, slots=True)
class MetricValue:
    """One named measurement, or an explicit reason it is unavailable.

    Attributes:
        name: Canonical metric name.
        value: Finite measured value, or None when unavailable.
        status: ``AVAILABLE`` or ``UNAVAILABLE``.
        reason: Why the measurement is unavailable.
    """

    name: str
    value: float | None
    status: Literal["AVAILABLE", "UNAVAILABLE"]
    reason: str | None = field(default=None)


@dataclass(frozen=True, slots=True)
class SplitEvidence:
    """Evidence for one task and split of one dataset.

    Attributes:
        task: ``classifier`` or ``segmentor``.
        split: ``train``, ``val``, or ``test``.
        dataset_hash: Content hash of the source dataset.
        metrics: Aggregate metric records for the split.
    """

    task: Literal["classifier", "segmentor"]
    split: Literal["train", "val", "test"]
    dataset_hash: str
    metrics: tuple[MetricValue, ...] = field(default=())
