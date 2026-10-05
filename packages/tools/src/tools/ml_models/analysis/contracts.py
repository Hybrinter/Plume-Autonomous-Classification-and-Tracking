"""Strict typed evidence records shared by evaluation, capture, and rendering.

These frozen records are the alignment keys, identity, and metric vocabulary
for the evidence-first analysis stack. All records validate at construction:
hashes are lowercase SHA-256 hex, counts are exact integers, and unavailable
measurements carry explicit reasons.

Contains:
  - SampleKey: dataset/split/shard/row identity for one scored sample.
  - NamedCount, MetricSupport: support units and named population counts.
  - ConfidenceInterval, MetricValue: one named measurement or an explicit
    unavailable state.
  - CurveEvidence: named finite coordinates with method metadata.
  - ArtifactRef: content-addressed reference to a bundle file.
  - DatasetIdentity, CodeIdentity, CheckpointIdentity: provenance records.
  - AvailabilityRecord: explicit output status with reasons.
  - SplitEvidence: task/split/dataset identity plus metrics and artifacts.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import re
from dataclasses import field
from typing import Annotated, Literal

from pydantic import (
    BeforeValidator,
    ConfigDict,
    FiniteFloat,
    StrictBool,
    StrictInt,
    model_validator,
)
from pydantic.dataclasses import dataclass

type Task = Literal["classifier", "segmentor"]
type Split = Literal["train", "val", "test"]
type SupportUnit = Literal["IMAGE", "PIXEL", "COMPONENT", "GROUP"]
type MetricStatus = Literal["AVAILABLE", "UNAVAILABLE"]
type CurveMethod = Literal["EXACT", "HISTOGRAM"]
type ArtifactKind = Literal[
    "TABLE", "CURVE", "FIGURE", "VISUAL", "PREDICTIONS", "CONFIG", "REFERENCE"
]
type AvailabilityStatus = Literal["AVAILABLE", "UNAVAILABLE", "SKIPPED"]


def _strict_finite(value: object) -> object:
    """Reject bools, strings, and other non-numeric scalars."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"expected a finite number; got {value!r}")
    return value


type FiniteNumber = Annotated[FiniteFloat, BeforeValidator(_strict_finite)]

_SCHEMA = ConfigDict(extra="forbid")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SNAKE_RE = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")


def is_sha256(value: str) -> bool:
    """Return True when ``value`` is a lowercase 64-hex SHA-256 string."""
    return _SHA256_RE.fullmatch(value) is not None


def _require_sha256(value: str, name: str) -> None:
    if not is_sha256(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex string; got {value!r}")


def _require_nonblank(value: str, name: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must be nonblank")


_RESERVED_STEMS = frozenset(
    ("CON", "PRN", "AUX", "NUL")
    + tuple(f"COM{i}" for i in range(1, 10))
    + tuple(f"LPT{i}" for i in range(1, 10))
)


def check_bundle_path(path: str) -> None:
    """Validate a safe relative POSIX bundle path.

    Args:
        path: Bundle-relative file path.

    Raises:
        ValueError: When the path is empty, absolute, drive-qualified, uses
            backslashes or colons, contains control characters, dot/empty/
            traversal components, components ending in a dot or space, or
            Windows reserved device stems.
    """
    if not path:
        raise ValueError("bundle path must be nonempty")
    if "\\" in path:
        raise ValueError(f"bundle path must use POSIX separators; got {path!r}")
    if path.startswith("/"):
        raise ValueError(f"bundle path must be relative; got {path!r}")
    if ":" in path:
        raise ValueError(f"bundle path must not contain a colon; got {path!r}")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in path):
        raise ValueError(f"bundle path contains a control character; got {path!r}")
    parts = path.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"bundle path has an unsafe component; got {path!r}")
    for part in parts:
        if part.endswith((".", " ")):
            raise ValueError(f"bundle path component ends in a dot or space; got {path!r}")
        if part.split(".")[0].upper() in _RESERVED_STEMS:
            raise ValueError(f"bundle path uses a reserved device name; got {path!r}")


def _check_unique_names(names: tuple[str, ...], what: str) -> None:
    if len(set(names)) != len(names):
        raise ValueError(f"{what} names must be unique")


@dataclass(frozen=True, slots=True, config=_SCHEMA)
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
    task: Task
    split: Split
    spatial_shard: tuple[StrictInt, StrictInt]
    row_index: StrictInt
    tile_id: str
    element: str

    @model_validator(mode="after")
    def _bounds(self) -> SampleKey:
        if not is_sha256(self.dataset_hash):
            raise ValueError("dataset_hash must be a lowercase SHA-256 hex string")
        if self.spatial_shard[0] < 1 or self.spatial_shard[1] < 1:
            raise ValueError("spatial_shard dimensions must be positive")
        if self.row_index < 0:
            raise ValueError("row_index must be nonnegative")
        _require_nonblank(self.tile_id, "tile_id")
        _require_nonblank(self.element, "element")
        return self


@dataclass(frozen=True, slots=True, config=_SCHEMA)
class NamedCount:
    """One named subpopulation count inside a support record.

    Attributes:
        name: Count label, unique within its support record.
        value: Nonnegative exact count.
    """

    name: str
    value: StrictInt

    @model_validator(mode="after")
    def _bounds(self) -> NamedCount:
        _require_nonblank(self.name, "name")
        if self.value < 0:
            raise ValueError("count value must be nonnegative")
        return self


@dataclass(frozen=True, slots=True, config=_SCHEMA)
class MetricSupport:
    """Population unit, total size, and named subpopulation counts.

    Attributes:
        unit: Measurement unit (``IMAGE``, ``PIXEL``, ``COMPONENT``, ``GROUP``).
        n: Number of measured units.
        counts: Named subpopulation counts, unique by name.
    """

    unit: SupportUnit
    n: StrictInt
    counts: tuple[NamedCount, ...] = field(default=())

    @model_validator(mode="after")
    def _bounds(self) -> MetricSupport:
        if self.n < 0:
            raise ValueError("support n must be nonnegative")
        _check_unique_names(tuple(c.name for c in self.counts), "support count")
        return self


@dataclass(frozen=True, slots=True, config=_SCHEMA)
class ConfidenceInterval:
    """A finite interval estimate with replicate metadata.

    Attributes:
        lower: Lower endpoint.
        upper: Upper endpoint.
        confidence: Nominal confidence, strictly inside ``(0, 1)``.
        method: Interval estimator name.
        n_replicates: Replicate budget.
        n_valid: Replicates that produced a usable estimate.
        seed: Replicate seed.
    """

    lower: FiniteNumber
    upper: FiniteNumber
    confidence: FiniteNumber = field(default=0.95)
    method: str = field(default="percentile_group_bootstrap")
    n_replicates: StrictInt = field(default=1000)
    n_valid: StrictInt = field(default=1000)
    seed: StrictInt = field(default=0)

    @model_validator(mode="after")
    def _bounds(self) -> ConfidenceInterval:
        if self.lower > self.upper:
            raise ValueError("confidence interval endpoints must be ordered")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must lie strictly inside (0, 1)")
        _require_nonblank(self.method, "method")
        if self.n_replicates < 1:
            raise ValueError("n_replicates must be positive")
        if not 0 <= self.n_valid <= self.n_replicates:
            raise ValueError("n_valid must lie in 0..n_replicates")
        return self


@dataclass(frozen=True, slots=True, config=_SCHEMA)
class MetricValue:
    """One named measurement, or an explicit reason it is unavailable.

    Attributes:
        name: Canonical lower-snake-case metric name.
        value: Finite measured value, or None when unavailable.
        status: ``AVAILABLE`` or ``UNAVAILABLE``.
        reason: Why the measurement is unavailable.
        unit: Physical/semantic unit of the value.
        aggregation: How per-unit values were combined.
        support: Population and unit information for the measurement.
        threshold: Operating threshold the value was measured at, if any.
        interval: Interval estimate for the value, if any.
    """

    name: str
    value: FiniteNumber | None
    status: MetricStatus
    reason: str | None = field(default=None)
    unit: str = field(default="dimensionless")
    aggregation: str = field(default="per_image_mean")
    support: MetricSupport = field(default_factory=lambda: MetricSupport(unit="IMAGE", n=0))
    threshold: FiniteNumber | None = field(default=None)
    interval: ConfidenceInterval | None = field(default=None)

    @model_validator(mode="after")
    def _bounds(self) -> MetricValue:
        if _SNAKE_RE.fullmatch(self.name) is None:
            raise ValueError(f"metric name must be lower snake case; got {self.name!r}")
        _require_nonblank(self.unit, "unit")
        _require_nonblank(self.aggregation, "aggregation")
        if self.status == "AVAILABLE":
            if self.value is None:
                raise ValueError("an AVAILABLE metric requires a finite value")
            if self.reason is not None:
                raise ValueError("an AVAILABLE metric cannot carry an unavailable reason")
        else:
            if self.value is not None:
                raise ValueError("an UNAVAILABLE metric cannot carry a value")
            if self.reason is None or not self.reason.strip():
                raise ValueError("an UNAVAILABLE metric requires a nonempty reason")
            if self.interval is not None:
                raise ValueError("an UNAVAILABLE metric cannot carry an interval")
        return self


@dataclass(frozen=True, slots=True, config=_SCHEMA)
class CurveEvidence:
    """Named finite coordinate series with explicit method metadata.

    Attributes:
        name: Curve name, unique within its evidence record.
        x_name: Horizontal axis name.
        y_name: Vertical axis name.
        x: Finite coordinates; aligned with ``y``.
        y: Values; None marks a missing observation, never a sentinel.
        x_unit: Horizontal axis unit.
        y_unit: Vertical axis unit.
        support: Population that produced the curve.
        method: ``EXACT`` coordinates or a ``HISTOGRAM`` approximation.
        thresholds: Per-coordinate operating thresholds; None may mark the
            predict-none operating point. Empty or aligned with ``x``.
        n_bins: Histogram bin count; required for ``HISTOGRAM`` only.
        notes: Method notes; a histogram requires at least one.
    """

    name: str
    x_name: str
    y_name: str
    x: tuple[FiniteNumber, ...]
    y: tuple[FiniteNumber | None, ...]
    x_unit: str
    y_unit: str
    support: MetricSupport
    method: CurveMethod = field(default="EXACT")
    thresholds: tuple[FiniteNumber | None, ...] = field(default=())
    n_bins: StrictInt | None = field(default=None)
    notes: tuple[str, ...] = field(default=())

    @model_validator(mode="after")
    def _bounds(self) -> CurveEvidence:
        _require_nonblank(self.name, "name")
        _require_nonblank(self.x_name, "x_name")
        _require_nonblank(self.y_name, "y_name")
        _require_nonblank(self.x_unit, "x_unit")
        _require_nonblank(self.y_unit, "y_unit")
        if len(self.x) != len(self.y):
            raise ValueError("curve x and y must have equal length")
        if self.thresholds and len(self.thresholds) != len(self.x):
            raise ValueError("curve thresholds must be empty or aligned with x")
        if self.method == "HISTOGRAM":
            if self.n_bins is None or self.n_bins < 2:
                raise ValueError("a histogram curve requires n_bins >= 2")
            if not any(note.strip() for note in self.notes):
                raise ValueError("a histogram curve requires a note describing it")
        elif self.n_bins is not None:
            raise ValueError("an EXACT curve cannot carry n_bins")
        return self


@dataclass(frozen=True, slots=True, config=_SCHEMA)
class ArtifactRef:
    """Content-addressed reference to one bundle file.

    Attributes:
        path: Safe relative POSIX path inside the bundle.
        sha256: Lowercase SHA-256 of the file bytes.
        size_bytes: Exact byte length.
        kind: Artifact category.
        format: Encoding/format tag (for example ``csv`` or ``parquet``).
        rows: Row count for tabular artifacts, if applicable.
        population: Population label the artifact covers, if applicable.
    """

    path: str
    sha256: str
    size_bytes: StrictInt
    kind: ArtifactKind
    format: str
    rows: StrictInt | None = field(default=None)
    population: str | None = field(default=None)

    @model_validator(mode="after")
    def _bounds(self) -> ArtifactRef:
        check_bundle_path(self.path)
        _require_sha256(self.sha256, "sha256")
        if self.size_bytes < 0:
            raise ValueError("size_bytes must be nonnegative")
        if self.rows is not None and self.rows < 0:
            raise ValueError("rows must be nonnegative")
        _require_nonblank(self.format, "format")
        if self.population is not None:
            _require_nonblank(self.population, "population")
        return self


@dataclass(frozen=True, slots=True, config=_SCHEMA)
class DatasetIdentity:
    """Verified identity of one finished dataset.

    Attributes:
        content_hash: SHA-256 over the shard files, excluding ``dataset.json``.
        manifest_hash: SHA-256 of the actual ``dataset.json`` bytes.
        schema_version: Manifest schema version.
        source: Source name from the manifest.
        band_names: Channel names, unique.
        gsd_reference_m: Reference used to encode model GSD.
    """

    content_hash: str
    manifest_hash: str
    schema_version: StrictInt
    source: str
    band_names: tuple[str, ...]
    gsd_reference_m: FiniteNumber

    @model_validator(mode="after")
    def _bounds(self) -> DatasetIdentity:
        _require_sha256(self.content_hash, "content_hash")
        _require_sha256(self.manifest_hash, "manifest_hash")
        if self.schema_version < 1:
            raise ValueError("schema_version must be positive")
        _require_nonblank(self.source, "source")
        if not self.band_names:
            raise ValueError("band_names must be nonempty")
        _check_unique_names(self.band_names, "band")
        for band in self.band_names:
            _require_nonblank(band, "band name")
        if self.gsd_reference_m <= 0.0:
            raise ValueError("gsd_reference_m must be finite and > 0")
        return self


@dataclass(frozen=True, slots=True, config=_SCHEMA)
class CodeIdentity:
    """Code provenance; unknown state is explicit, never fabricated.

    Attributes:
        revision: Commit/revision identifier, or None when unknown.
        dirty: Whether the working tree was dirty, or None when unknown.
        diff_hash: Hash of uncommitted changes, if computed.
        reason: Why revision/dirty are unknown; required when either is None.
    """

    revision: str | None
    dirty: StrictBool | None
    diff_hash: str | None = field(default=None)
    reason: str | None = field(default=None)

    @model_validator(mode="after")
    def _bounds(self) -> CodeIdentity:
        if self.revision is not None:
            _require_nonblank(self.revision, "revision")
        if self.diff_hash is not None:
            _require_sha256(self.diff_hash, "diff_hash")
        if (self.revision is None or self.dirty is None) and (
            self.reason is None or not self.reason.strip()
        ):
            raise ValueError("absent revision or dirty state requires a reason")
        return self


@dataclass(frozen=True, slots=True, config=_SCHEMA)
class CheckpointIdentity:
    """Identity of the checkpoint an analysis measured.

    Attributes:
        sha256: Lowercase SHA-256 of the checkpoint file.
        kind: ``classifier`` or ``segmentor``.
        arch: Architecture name.
        training_dataset_hash: Content hash of the training dataset.
        epoch: Epoch index, or None when unknown.
        step: Step index, or None when unknown.
    """

    sha256: str
    kind: Task
    arch: str
    training_dataset_hash: str
    epoch: StrictInt | None = field(default=None)
    step: StrictInt | None = field(default=None)

    @model_validator(mode="after")
    def _bounds(self) -> CheckpointIdentity:
        _require_sha256(self.sha256, "sha256")
        _require_nonblank(self.arch, "arch")
        _require_sha256(self.training_dataset_hash, "training_dataset_hash")
        if self.epoch is not None and self.epoch < 0:
            raise ValueError("epoch must be nonnegative")
        if self.step is not None and self.step < 0:
            raise ValueError("step must be nonnegative")
        return self


@dataclass(frozen=True, slots=True, config=_SCHEMA)
class AvailabilityRecord:
    """Explicit availability of one optional or required output.

    Attributes:
        name: Output name, unique within its summary.
        status: ``AVAILABLE``, ``UNAVAILABLE``, or ``SKIPPED``.
        reason: Why the output is absent; required unless ``AVAILABLE``.
        required: Whether the output is required for a complete summary.
    """

    name: str
    status: AvailabilityStatus
    reason: str | None = field(default=None)
    required: StrictBool = field(default=False)

    @model_validator(mode="after")
    def _bounds(self) -> AvailabilityRecord:
        _require_nonblank(self.name, "name")
        if self.status == "AVAILABLE":
            if self.reason is not None:
                raise ValueError("an AVAILABLE output cannot carry a reason")
        elif self.reason is None or not self.reason.strip():
            raise ValueError("an UNAVAILABLE/SKIPPED output requires a reason")
        return self


@dataclass(frozen=True, slots=True, config=_SCHEMA)
class SplitEvidence:
    """Evidence for one task and split of one dataset.

    Attributes:
        task: ``classifier`` or ``segmentor``.
        split: ``train``, ``val``, or ``test``.
        dataset_hash: Content hash of the source dataset.
        metrics: Aggregate metric records, unique by name.
        checkpoint_hash: Hash of the measured checkpoint, if any.
        dataset_manifest_hash: Hash of the source ``dataset.json``, if any.
        curves: Curve records, unique by name.
        artifacts: Artifact references, unique by path.
        support: Population that produced the metrics.
        warnings: Non-fatal caveats.
    """

    task: Task
    split: Split
    dataset_hash: str
    metrics: tuple[MetricValue, ...] = field(default=())
    checkpoint_hash: str | None = field(default=None)
    dataset_manifest_hash: str | None = field(default=None)
    curves: tuple[CurveEvidence, ...] = field(default=())
    artifacts: tuple[ArtifactRef, ...] = field(default=())
    support: MetricSupport = field(default_factory=lambda: MetricSupport(unit="IMAGE", n=0))
    warnings: tuple[str, ...] = field(default=())

    @model_validator(mode="after")
    def _bounds(self) -> SplitEvidence:
        _require_sha256(self.dataset_hash, "dataset_hash")
        if self.checkpoint_hash is not None:
            _require_sha256(self.checkpoint_hash, "checkpoint_hash")
        if self.dataset_manifest_hash is not None:
            _require_sha256(self.dataset_manifest_hash, "dataset_manifest_hash")
        _check_unique_names(tuple(m.name for m in self.metrics), "metric")
        _check_unique_names(tuple(c.name for c in self.curves), "curve")
        _check_unique_names(tuple(a.path for a in self.artifacts), "artifact path")
        return self
