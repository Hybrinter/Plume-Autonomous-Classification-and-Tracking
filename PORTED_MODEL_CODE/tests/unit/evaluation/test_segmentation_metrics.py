"""Tests for segmentation plotting helpers."""

from __future__ import annotations

from pathlib import Path

from smoke_detection.evaluation.segmentation_metrics import (
    plot_area_ratio_distribution,
    plot_iou_distribution,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _is_png(path: Path) -> bool:
    return path.read_bytes()[:8] == PNG_MAGIC


def test_plot_iou_distribution(tmp_path):
    out = tmp_path / "iou.png"
    plot_iou_distribution([0.1, 0.5, 0.9, 0.3, 0.7], out)
    assert out.exists() and _is_png(out)


def test_plot_iou_distribution_empty(tmp_path):
    out = tmp_path / "iou_empty.png"
    plot_iou_distribution([], out)
    assert out.exists() and _is_png(out)


def test_plot_area_ratio_distribution(tmp_path):
    out = tmp_path / "ar.png"
    plot_area_ratio_distribution([0.5, 1.0, 1.5, 2.0], out)
    assert out.exists() and _is_png(out)


def test_plot_area_ratio_distribution_empty(tmp_path):
    out = tmp_path / "ar_empty.png"
    plot_area_ratio_distribution([], out)
    assert out.exists() and _is_png(out)
