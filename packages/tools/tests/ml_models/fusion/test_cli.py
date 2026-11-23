"""Tests for the fusion-trade command line."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tools.ml_models.fusion.cli import main


def _capture(path: Path, rows: list[tuple[str, float, float]]) -> None:
    """Write capture rows of (bin, label, logit)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({"bin_id": b, "label": y, "metrics": [{"name": "logit", "value": v}]})
        for b, y, v in rows
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_thresholds_command(tmp_path: Path) -> None:
    """thresholds writes JSON rows and a plot from analysis capture rows."""
    rows = [("el05", 1.0, 2.0), ("el05", 0.0, -2.0), ("el45", 1.0, -1.0), ("el45", 0.0, -3.0)]
    _capture(tmp_path / "a" / "capture" / "val" / "rows.jsonl", rows)
    _capture(tmp_path / "a" / "capture" / "test" / "rows.jsonl", rows)
    out, plot = tmp_path / "thr.json", tmp_path / "thr.png"
    args = ["thresholds", "--analysis", str(tmp_path / "a"), "--out", str(out)]
    assert main([*args, "--plot", str(plot)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert [r["bin_id"] for r in payload["rows"]] == ["el05", "el45", "all"]
    assert payload["rows"][1]["f1_per_bin"] == 1.0
    assert plot.exists()


def test_seed_plot_command(tmp_path: Path) -> None:
    """seed-plot reads the test metric from each JSON file."""
    files = []
    for i, ap in enumerate((0.7, 0.8)):
        path = tmp_path / f"s{i}.json"
        path.write_text(json.dumps({"test": {"average_precision": ap}}), encoding="utf-8")
        files.append(str(path))
    out = tmp_path / "seeds.png"
    assert main(["seed-plot", "--entry", "Early=" + ",".join(files), "--out", str(out)]) == 0
    assert out.exists()
    with pytest.raises(SystemExit):
        main(["seed-plot", "--entry", "Early", "--out", str(out)])
