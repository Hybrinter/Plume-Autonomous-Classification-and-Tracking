"""Whole-cohort deterministic prediction gallery selection over frozen scalar rows.

Selections use the same seed/key priority and failure ordering as bounded
capture. High-confidence classifier errors are globally selected from all
errors by unweighted BCE, monotone in wrong predicted-class confidence.
Missing cache entries remain missing rather than replacing chosen failures
with easier examples. No image or model is read here.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from flight.libs.types import Err, Ok, Result

from tools.ml_models.analysis.capture import CaptureRow
from tools.ml_models.analysis.config import CaptureConfig
from tools.ml_models.analysis.contracts import (
    AvailabilityRecord,
    AvailabilityStatus,
    SplitEvidence,
)
from tools.ml_models.analysis.model_figures import captured_values, validate_figure_rows


@dataclass(frozen=True, slots=True)
class PredictionGallery:
    """Selected complete-cohort identities, not a statistically representative claim."""

    identifier: str
    family: str
    rows: tuple[CaptureRow, ...]
    availability: AvailabilityRecord
    selection_method: str


def prediction_key(row: CaptureRow) -> str:
    """Return the content key used by existing bounded prediction capture."""
    raw = json.dumps(
        asdict(row.key), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def prediction_priority(row: CaptureRow, seed: int) -> tuple[str, str]:
    """Return the exact capture seed/key tie priority, independent of traversal order."""
    raw = json.dumps(
        {"seed": seed, "key": asdict(row.key)},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(raw).hexdigest(), prediction_key(row)


def prediction_gallery_data(
    evidence: SplitEvidence,
    rows: tuple[CaptureRow, ...],
    cfg: CaptureConfig,
) -> Result[tuple[PredictionGallery, ...], str]:
    """Freeze representatives and global error families; capture limits do not tune scores."""
    checked = validate_figure_rows(evidence, rows)
    if isinstance(checked, Err):
        return checked
    if evidence.task == "classifier" and any(
        captured_values(row).get("binary_cross_entropy") != row.failure_score for row in rows
    ):
        return Err("classifier failure ordering must match the frozen unweighted BCE")
    representative = sorted(rows, key=lambda row: prediction_priority(row, cfg.seed))
    failure = sorted(
        rows, key=lambda row: (-row.failure_score, *prediction_priority(row, cfg.seed))
    )
    families: tuple[tuple[str, str, list[CaptureRow]], ...] = (
        (
            "representative",
            "Seeded whole-cohort display selection; not independent/random sampling",
            representative,
        ),
        (
            "false_positive",
            "Whole-cohort false positives, descending captured failure score",
            [row for row in failure if row.false_positive],
        ),
        (
            "false_negative",
            "Whole-cohort false negatives, descending captured failure score",
            [row for row in failure if row.false_negative],
        ),
    )
    if evidence.task == "classifier":
        families += (
            (
                "high_confidence_error",
                "Whole-cohort errors, descending captured unweighted BCE "
                "(monotone wrong-class confidence)",
                [row for row in failure if row.false_positive or row.false_negative],
            ),
        )
    else:
        families += (
            (
                "worst",
                "Whole-cohort descending captured failure score; not a representative sample",
                failure,
            ),
        )
    galleries: list[PredictionGallery] = []
    for family, method, candidates in families:
        disabled = cfg.max_preview_images == 0 or cfg.examples_per_family == 0
        selected = tuple(candidates[: cfg.examples_per_family]) if not disabled else ()
        status: AvailabilityStatus = (
            "SKIPPED" if disabled else "AVAILABLE" if selected else "UNAVAILABLE"
        )
        reason = (
            "Prediction preview capture disabled"
            if disabled
            else None
            if selected
            else "No eligible captured rows for this gallery family"
        )
        galleries.append(
            PredictionGallery(
                evidence.split + "_" + family,
                family,
                selected,
                AvailabilityRecord(
                    name="prediction_visual:" + evidence.split + ":" + family,
                    status=status,
                    reason=reason,
                ),
                method,
            )
        )
    return Ok(tuple(galleries))
