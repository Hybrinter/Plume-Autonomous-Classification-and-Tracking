"""Tests for validation-chosen classifier thresholds."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from tools.ml_models.fusion.thresholds import (
    ALL_BINS,
    ScoredRows,
    best_threshold,
    choose_thresholds,
    f1_score,
    read_capture_rows,
    score_thresholds,
)


def _rows(logit: list[float], label: list[bool], bins: list[str]) -> ScoredRows:
    """Build scored rows."""
    return ScoredRows(
        logit=np.asarray(logit, dtype=np.float64),
        label=np.asarray(label, dtype=np.bool_),
        bin_id=np.asarray(bins, dtype=np.str_),
    )


def test_f1_score_counts() -> None:
    """F1 matches 2TP / (2TP + FP + FN) and is 0 with no positives."""
    pred = np.array([True, True, False, False])
    label = np.array([True, False, True, False])
    assert f1_score(pred, label) == pytest.approx(0.5)
    none = np.zeros(3, dtype=np.bool_)
    assert f1_score(none, none) == 0.0


def test_best_threshold_separates_classes() -> None:
    """A separable split gets a threshold between the two classes."""
    logit = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    label = logit > 0
    t = best_threshold(logit, label)
    assert -1.0 < t <= 1.0
    assert f1_score(logit >= t, label) == 1.0
    with pytest.raises(ValueError):
        best_threshold(np.array([], dtype=np.float64), np.array([], dtype=np.bool_))


def test_per_bin_threshold_beats_fixed_on_shifted_bin() -> None:
    """A bin whose scores are shifted low recovers F1 with its own threshold."""
    val = _rows(
        [-4.0, -3.0, 1.0, 2.0, -6.0, -5.5, -2.5, -2.0],
        [False, False, True, True, False, False, True, True],
        ["el05"] * 4 + ["el45"] * 4,
    )
    choice = choose_thresholds(val)
    assert set(choice.per_bin_logit) == {"el05", "el45"}
    rows = score_thresholds(val, choice)
    assert [r.bin_id for r in rows] == ["el05", "el45", ALL_BINS]
    el45 = rows[1]
    assert el45.f1_fixed == 0.0
    assert el45.f1_per_bin == 1.0
    assert rows[-1].n == 8


def test_unseen_test_bin_uses_global_threshold() -> None:
    """A test bin without validation rows falls back to the global threshold."""
    val = _rows([-1.0, 1.0], [False, True], ["el05", "el05"])
    test = _rows([-1.0, 1.0], [False, True], ["el25", "el25"])
    rows = score_thresholds(test, choose_thresholds(val))
    assert rows[0].f1_per_bin == rows[0].f1_global


def test_read_capture_rows(tmp_path: Path) -> None:
    """Capture rows give the logit metric, label, and bin of each row."""
    path = tmp_path / "rows.jsonl"
    rows = [
        {"bin_id": "el05", "label": 1.0, "metrics": [{"name": "logit", "value": 2.5}]},
        {"bin_id": "el45", "label": 0.0, "metrics": [{"name": "logit", "value": -1.0}]},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    scored = read_capture_rows(path)
    assert scored.logit.tolist() == [2.5, -1.0]
    assert scored.label.tolist() == [True, False]
    assert scored.bin_id.tolist() == ["el05", "el45"]
    path.write_text(json.dumps({"bin_id": "el05", "label": 1.0, "metrics": []}), encoding="utf-8")
    with pytest.raises(ValueError):
        read_capture_rows(path)
