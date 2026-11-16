"""Classifier decision thresholds chosen on validation and scored on test.

Contains:
  - ScoredRows: logits, labels, and look-angle bins of one split.
  - ThresholdChoice: one global logit threshold and one per look-angle bin.
  - ThresholdRow: test F1 of one bin at the fixed, global, and per-bin thresholds.
  - read_capture_rows: read a capture ``rows.jsonl`` into ScoredRows.
  - f1_score: F1 of boolean predictions.
  - best_threshold: logit threshold with the highest F1.
  - choose_thresholds: global and per-bin thresholds from validation rows.
  - score_thresholds: test F1 per bin and for all rows.

Thresholds are logits. The fixed threshold is logit 0 (probability 0.5). The
candidate grid is the 1st to 99th percentile of the split's own logits, so it
follows the model's score scale. A test bin that has no validation rows uses
the global threshold.

Satisfies: REQ-AIML-COMP-001.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

ALL_BINS = "all"
_GRID_POINTS = 197


@dataclass(frozen=True, slots=True)
class ScoredRows:
    """Classifier scores of one split.

    Attributes:
        logit: Model logit per tile.  # np.ndarray[float64, (N,)]
        label: True when the tile holds a plume.  # np.ndarray[bool, (N,)]
        bin_id: Look-angle bin per tile, such as ``el45``.  # np.ndarray[str, (N,)]
    """

    logit: npt.NDArray[np.float64]
    label: npt.NDArray[np.bool_]
    bin_id: npt.NDArray[np.str_]


@dataclass(frozen=True, slots=True)
class ThresholdChoice:
    """Validation-chosen logit thresholds.

    Attributes:
        global_logit: One threshold for every tile.
        per_bin_logit: One threshold per look-angle bin.
    """

    global_logit: float
    per_bin_logit: dict[str, float]


@dataclass(frozen=True, slots=True)
class ThresholdRow:
    """Test F1 of one look-angle bin, or of all rows.

    Attributes:
        bin_id: Bin name, or ``all``.
        n: Tile count.
        f1_fixed: F1 at probability 0.5.
        f1_global: F1 at the global threshold.
        f1_per_bin: F1 at each tile's bin threshold.
    """

    bin_id: str
    n: int
    f1_fixed: float
    f1_global: float
    f1_per_bin: float


def read_capture_rows(path: Path) -> ScoredRows:
    """Read the logit, label, and bin of each row of a capture ``rows.jsonl``.

    Args:
        path: Capture rows file written by the model analysis.

    Returns:
        ScoredRows: One entry per row, in file order.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If a row has no ``logit`` metric.
    """
    logit: list[float] = []
    label: list[bool] = []
    bins: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            values = [m["value"] for m in row["metrics"] if m["name"] == "logit"]
            if not values:
                raise ValueError(f"capture row without a logit in {path}")
            logit.append(float(values[0]))
            label.append(float(row["label"]) > 0.5)
            bins.append(str(row["bin_id"]))
    return ScoredRows(
        logit=np.asarray(logit, dtype=np.float64),
        label=np.asarray(label, dtype=np.bool_),
        bin_id=np.asarray(bins, dtype=np.str_),
    )


def f1_score(pred: npt.NDArray[np.bool_], label: npt.NDArray[np.bool_]) -> float:
    """Return the F1 score of boolean predictions.

    Args:
        pred: Predicted positive per tile.  # np.ndarray[bool, (N,)]
        label: True positive per tile.  # np.ndarray[bool, (N,)]

    Returns:
        float: ``2 TP / (2 TP + FP + FN)``; 0 when there are no positives at all.
    """
    tp = int(np.sum(pred & label))
    fp = int(np.sum(pred & ~label))
    fn = int(np.sum(~pred & label))
    return 2.0 * tp / max(2 * tp + fp + fn, 1)


def best_threshold(logit: npt.NDArray[np.float64], label: npt.NDArray[np.bool_]) -> float:
    """Return the logit threshold with the highest F1.

    Args:
        logit: Model logit per tile.  # np.ndarray[float64, (N,)]
        label: True positive per tile.  # np.ndarray[bool, (N,)]

    Returns:
        float: The best grid threshold. Ties keep the lowest threshold.

    Raises:
        ValueError: If ``logit`` is empty.
    """
    if logit.size == 0:
        raise ValueError("best_threshold needs at least one row")
    grid = np.quantile(logit, np.linspace(0.01, 0.99, _GRID_POINTS))
    scores = [f1_score(logit >= t, label) for t in grid]
    return float(grid[int(np.argmax(scores))])


def choose_thresholds(val: ScoredRows) -> ThresholdChoice:
    """Choose one global and one per-bin threshold on validation rows.

    Args:
        val: Validation split scores.

    Returns:
        ThresholdChoice: The selected logit thresholds.
    """
    per_bin = {
        str(b): best_threshold(val.logit[val.bin_id == b], val.label[val.bin_id == b])
        for b in np.unique(val.bin_id)
    }
    return ThresholdChoice(global_logit=best_threshold(val.logit, val.label), per_bin_logit=per_bin)


def score_thresholds(test: ScoredRows, choice: ThresholdChoice) -> tuple[ThresholdRow, ...]:
    """Score test F1 at the fixed, global, and per-bin thresholds.

    Args:
        test: Test split scores.
        choice: Thresholds chosen on validation.

    Returns:
        tuple[ThresholdRow, ...]: One row per test bin in sorted order, then ``all``.
    """
    per_tile = np.asarray(
        [choice.per_bin_logit.get(str(b), choice.global_logit) for b in test.bin_id],
        dtype=np.float64,
    )
    rows: list[ThresholdRow] = []
    for bin_id in [*(str(b) for b in np.unique(test.bin_id)), ALL_BINS]:
        mask = np.ones(test.label.shape, dtype=np.bool_)
        if bin_id != ALL_BINS:
            mask = test.bin_id == bin_id
        logit, label = test.logit[mask], test.label[mask]
        rows.append(
            ThresholdRow(
                bin_id=bin_id,
                n=int(mask.sum()),
                f1_fixed=f1_score(logit >= 0.0, label),
                f1_global=f1_score(logit >= choice.global_logit, label),
                f1_per_bin=f1_score(logit >= per_tile[mask], label),
            )
        )
    return tuple(rows)
