"""Tagged, versioned summary records for dataset and model analyses.

Each analysis execution produces exactly one of these two records. Both are
tagged by ``summary_kind`` and pinned to ``schema_version`` 1; unknown kinds
and versions are rejected rather than coerced. Identity fields are validated
against their evidence: a split that disagrees with the summary's dataset or
checkpoint is a construction error, not a normalization opportunity.

Contains:
  - DatasetSummary: evidence of one dataset-analysis execution.
  - ModelTrainingSummary: evidence of one model/training-analysis execution.
  - Summary: the tagged union consumed by codecs.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

from dataclasses import field
from typing import Literal

from pydantic import ConfigDict, StrictInt, model_validator
from pydantic.dataclasses import dataclass

from tools.ml_models.analysis.contracts import (
    ArtifactRef,
    AvailabilityRecord,
    CheckpointIdentity,
    CodeIdentity,
    DatasetIdentity,
    MetricValue,
    SplitEvidence,
    is_sha256,
)

type SummaryStatus = Literal["COMPLETE", "PARTIAL", "FAILED"]
SUMMARY_SCHEMA_VERSION = 1
_SCHEMA = ConfigDict(extra="forbid")


def _check_digest(value: str, name: str) -> None:
    if not is_sha256(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex string; got {value!r}")


def _check_unique(values: tuple[str, ...], what: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{what} must be unique")


def _check_status(status: SummaryStatus, outputs: tuple[AvailabilityRecord, ...]) -> None:
    if status == "COMPLETE":
        for output in outputs:
            if output.required and output.status != "AVAILABLE":
                raise ValueError(
                    f"a COMPLETE summary cannot mark required output "
                    f"{output.name!r} {output.status}"
                )


def _check_common(
    measurement_id: str,
    config_digest: str,
    schema_version: int,
    metrics: tuple[MetricValue, ...],
    splits: tuple[SplitEvidence, ...],
    artifacts: tuple[ArtifactRef, ...],
    outputs: tuple[AvailabilityRecord, ...],
    status: SummaryStatus,
) -> None:
    _check_digest(measurement_id, "measurement_id")
    _check_digest(config_digest, "config_digest")
    if schema_version != SUMMARY_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported summary schema_version {schema_version}; "
            f"expected {SUMMARY_SCHEMA_VERSION}"
        )
    _check_unique(tuple(m.name for m in metrics), "metric names")
    _check_unique(tuple(f"{s.task}:{s.split}" for s in splits), "split identities")
    _check_unique(tuple(a.path for a in artifacts), "artifact paths")
    _check_unique(tuple(o.name for o in outputs), "output names")
    _check_status(status, outputs)


def _check_split_dataset(split: SplitEvidence, dataset: DatasetIdentity) -> None:
    if split.dataset_hash != dataset.content_hash:
        raise ValueError("split dataset_hash conflicts with the summary dataset identity")
    if (
        split.dataset_manifest_hash is not None
        and split.dataset_manifest_hash != dataset.manifest_hash
    ):
        raise ValueError("split dataset_manifest_hash conflicts with the summary dataset")


@dataclass(frozen=True, slots=True, config=_SCHEMA)
class DatasetSummary:
    """Evidence of one dataset-analysis execution.

    Attributes:
        measurement_id: Frozen scientific identity of the measurement.
        dataset: Verified identity of the analyzed dataset.
        code: Code provenance; unknown state is explicit.
        config_digest: Hash of the resolved scientific settings.
        metrics: Whole-dataset metric records, unique by name.
        splits: Per-split evidence, unique by task/split.
        artifacts: Bundle file references, unique by path.
        outputs: Availability of declared outputs, unique by name.
        warnings: Non-fatal caveats.
        status: ``COMPLETE``, ``PARTIAL``, or ``FAILED``.
        schema_version: Summary schema; only 1 is supported.
        summary_kind: Tag for the canonical codec.
    """

    measurement_id: str
    dataset: DatasetIdentity
    code: CodeIdentity
    config_digest: str
    metrics: tuple[MetricValue, ...] = field(default=())
    splits: tuple[SplitEvidence, ...] = field(default=())
    artifacts: tuple[ArtifactRef, ...] = field(default=())
    outputs: tuple[AvailabilityRecord, ...] = field(default=())
    warnings: tuple[str, ...] = field(default=())
    status: SummaryStatus = field(default="COMPLETE")
    schema_version: StrictInt = field(default=SUMMARY_SCHEMA_VERSION)
    summary_kind: Literal["DATASET_ANALYSIS"] = field(default="DATASET_ANALYSIS")

    @model_validator(mode="after")
    def _bounds(self) -> DatasetSummary:
        _check_common(
            self.measurement_id,
            self.config_digest,
            self.schema_version,
            self.metrics,
            self.splits,
            self.artifacts,
            self.outputs,
            self.status,
        )
        for split in self.splits:
            _check_split_dataset(split, self.dataset)
            if split.checkpoint_hash is not None:
                raise ValueError("a dataset-analysis split cannot carry a checkpoint hash")
        return self


@dataclass(frozen=True, slots=True, config=_SCHEMA)
class ModelTrainingSummary:
    """Evidence of one model/training-analysis execution.

    Attributes:
        measurement_id: Frozen scientific identity of the measurement.
        training_dataset: Verified identity of the training dataset.
        evaluation_dataset: Verified identity of the evaluation dataset.
        checkpoint: Identity of the measured checkpoint.
        code: Code provenance; unknown state is explicit.
        config_digest: Hash of the resolved scientific settings.
        metrics: Run-level metric records, unique by name.
        splits: Per-split evidence, unique by task/split.
        artifacts: Bundle file references, unique by path.
        outputs: Availability of declared outputs, unique by name.
        warnings: Non-fatal caveats.
        history: Reference to the training history artifact, if captured.
        status: ``COMPLETE``, ``PARTIAL``, or ``FAILED``.
        schema_version: Summary schema; only 1 is supported.
        summary_kind: Tag for the canonical codec.
    """

    measurement_id: str
    training_dataset: DatasetIdentity
    evaluation_dataset: DatasetIdentity
    checkpoint: CheckpointIdentity
    code: CodeIdentity
    config_digest: str
    metrics: tuple[MetricValue, ...] = field(default=())
    splits: tuple[SplitEvidence, ...] = field(default=())
    artifacts: tuple[ArtifactRef, ...] = field(default=())
    outputs: tuple[AvailabilityRecord, ...] = field(default=())
    warnings: tuple[str, ...] = field(default=())
    history: ArtifactRef | None = field(default=None)
    status: SummaryStatus = field(default="COMPLETE")
    schema_version: StrictInt = field(default=SUMMARY_SCHEMA_VERSION)
    summary_kind: Literal["MODEL_TRAINING_ANALYSIS"] = field(default="MODEL_TRAINING_ANALYSIS")

    @model_validator(mode="after")
    def _bounds(self) -> ModelTrainingSummary:
        _check_common(
            self.measurement_id,
            self.config_digest,
            self.schema_version,
            self.metrics,
            self.splits,
            self.artifacts,
            self.outputs,
            self.status,
        )
        if self.checkpoint.training_dataset_hash != self.training_dataset.content_hash:
            raise ValueError(
                "checkpoint.training_dataset_hash conflicts with the training dataset identity"
            )
        for split in self.splits:
            _check_split_dataset(split, self.evaluation_dataset)
            if split.task != self.checkpoint.kind:
                raise ValueError("split task conflicts with the summary checkpoint kind")
            if (
                split.checkpoint_hash is not None
                and split.checkpoint_hash != self.checkpoint.sha256
            ):
                raise ValueError("split checkpoint_hash conflicts with the summary checkpoint")
        return self


type Summary = DatasetSummary | ModelTrainingSummary
