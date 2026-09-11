"""Write figures and a markdown summary into an inference run directory.

Contains:
  - write_report: emit figures/ PNGs and report.md from history and eval files.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import json
from pathlib import Path

from tools.inference.plots import (
    failure_figures,
    history_figures,
    overlay_figures,
    save_figures,
)


def write_report(run_dir: str | Path) -> Path:
    """Write figures and report.md into ``run_dir``.

    Args:
        run_dir: Training run directory with history.csv and optional eval.json.

    Returns:
        Path: Path of ``report.md``.

    Raises:
        FileNotFoundError: If the run directory is missing.
    """
    root = Path(run_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"run directory not found: {root}")
    figures_dir = root / "figures"
    figures = history_figures(root / "history.csv")
    figures.extend(overlay_figures(root / "predictions.npz"))
    figures.extend(failure_figures(root / "predictions.npz"))
    written = save_figures(figures, figures_dir)
    summary: dict[str, object] = {}
    summary_path = root / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    eval_payload: dict[str, object] = {}
    eval_path = root / "eval.json"
    if eval_path.is_file():
        eval_payload = json.loads(eval_path.read_text(encoding="utf-8"))
    lines = [
        f"# Inference run {root.name}",
        "",
        f"Kind: {summary.get('kind', '')}",
        f"Architecture: {summary.get('arch', '')}",
        f"Best epoch: {summary.get('best_epoch', '')}",
        f"Val metric: {summary.get('val_metric', '')} = {summary.get('best_val_metric', '')}",
        f"Dataset hash: {summary.get('dataset_hash', '')}",
        "",
    ]
    if eval_payload:
        lines.append("## Last eval")
        lines.append("")
        for key, value in eval_payload.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
    val_keys = [key for key in summary if str(key).startswith("val_")]
    test_keys = [key for key in summary if str(key).startswith("test_")]
    if val_keys:
        lines.append("## Val")
        lines.append("")
        for key in val_keys:
            lines.append(f"- {key}: {summary[key]}")
        lines.append("")
    if test_keys:
        lines.append("## Test")
        lines.append("")
        for key in test_keys:
            lines.append(f"- {key}: {summary[key]}")
        lines.append("")
    if written:
        lines.append("## Figures")
        lines.append("")
        for path in written:
            rel = path.relative_to(root).as_posix()
            lines.append(f"- [{path.stem}]({rel})")
        lines.append("")
    report_path = root / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
