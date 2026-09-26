"""Flight hit-rate frontier and study Dice frontier."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tools.ml_models.analysis.pareto import flight_pareto, study_pareto


def _write_run(root: Path, name: str, **fields: object) -> Path:
    """Write a minimal summary.json run directory."""
    dest = root / name
    dest.mkdir()
    payload: dict[str, object] = {
        "run_id": name,
        "kind": "segmentor",
        "arch": "dilatenet",
        "n_params": 100,
        "flops": 10,
    }
    payload.update(fields)
    (dest / "summary.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return dest


def test_flight_pareto_uses_hit_rate_and_study_uses_val_dice(tmp_path: Path) -> None:
    """A hit-rate summary is a flight point. A chip run is ranked on val Dice."""
    flight = _write_run(
        tmp_path,
        "flight",
        n_params=80,
        full_frame_hit_rate=0.7,
        val_mean_dice=0.1,
    )
    cheap = _write_run(tmp_path, "chip-small", n_params=40, val_mean_dice=0.4)
    large = _write_run(tmp_path, "chip-large", n_params=90, val_mean_dice=0.6)

    flight_front = flight_pareto((flight, cheap, large))
    study_front = study_pareto((flight, cheap, large))

    assert [point.run_id for point in flight_front] == ["flight"]
    assert flight_front[0].score == pytest.approx(0.7)
    assert flight_front[0].cost == pytest.approx(80)
    assert [point.run_id for point in study_front] == ["chip-small", "chip-large"]
    assert study_front[0].score == pytest.approx(0.4)
