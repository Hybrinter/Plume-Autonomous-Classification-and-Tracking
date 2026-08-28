"""Tests for inference report figures (torch-free)."""

from pathlib import Path

import numpy as np
from tools.inference.plots import history_figures, overlay_figures
from tools.inference.report import write_report


def test_history_figures_from_csv(tmp_path: Path) -> None:
    """history_figures draws train and val loss curves."""
    path = tmp_path / "history.csv"
    path.write_text(
        "epoch,split,loss,mean_iou\n1,train,0.8,0.1\n1,val,0.9,0.05\n",
        encoding="utf-8",
    )
    figures = history_figures(path)
    names = {item.name for item in figures}
    assert "curve_loss" in names
    assert "curve_mean_iou" in names


def test_write_report_with_history_only(tmp_path: Path) -> None:
    """write_report emits report.md and PNGs from history.csv."""
    (tmp_path / "history.csv").write_text(
        "epoch,split,loss\n1,train,0.4\n1,val,0.5\n",
        encoding="utf-8",
    )
    (tmp_path / "summary.json").write_text(
        '{"run_id": "x", "kind": "segmentor", "arch": "unet", "best_epoch": 1}\n',
        encoding="utf-8",
    )
    report = write_report(tmp_path)
    assert report.is_file()
    text = report.read_text(encoding="utf-8")
    assert "segmentor" in text
    assert (tmp_path / "figures" / "curve_loss.png").is_file()


def test_overlay_figures_from_predictions(tmp_path: Path) -> None:
    """overlay_figures builds a panel from a tiny predictions.npz."""
    images = np.zeros((2, 4, 8, 8), dtype=np.float32)
    images[0, 0, 2:6, 2:6] = 1.0
    masks = np.zeros((2, 1, 8, 8), dtype=np.float32)
    masks[0, 0, 2:6, 2:6] = 1.0
    logits = np.full((2, 1, 8, 8), -2.0, dtype=np.float32)
    logits[0, 0, 2:6, 2:6] = 4.0
    np.savez(
        tmp_path / "predictions.npz",
        images=images,
        targets=masks,
        logits=logits,
        scores=np.array([1.0, 0.0], dtype=np.float32),
        kind=np.array("segmentor"),
    )
    figures = overlay_figures(tmp_path / "predictions.npz")
    assert len(figures) == 1
    assert figures[0].name == "overlays"
