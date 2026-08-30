"""Finalize a trained run: test eval, ONNX export, and the acceptance gate."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from tools.inference.finalize import finalize
from tools.inference.train import TrainConfig, train

_HAS_ONNX = importlib.util.find_spec("onnx") is not None
_HAS_ORT = importlib.util.find_spec("onnxruntime") is not None
_skip_export = pytest.mark.skipif(
    not (_HAS_ONNX and _HAS_ORT), reason="onnx and onnxruntime extras not installed"
)


@_skip_export
def test_finalize_writes_eval_export_and_report(tmp_path: Path) -> None:
    """A one-epoch synthetic run produces eval.json, ONNX, and finalize.json."""
    run = train(
        TrainConfig(
            kind="segmentor",
            epochs=1,
            batch_size=2,
            synthetic_samples=8,
            input_height_px=32,
            input_width_px=32,
            run_dir=str(tmp_path / "runs"),
            seed=0,
        )
    )
    report = finalize(
        run,
        int8=False,
        scenes_limit=2,
        min_iou=0.0,
        max_latency_ms=10_000.0,
    )
    payload = json.loads((run / "finalize.json").read_text(encoding="utf-8"))
    assert Path(report.eval_path).is_file()
    assert Path(report.fp32_onnx).is_file()
    assert Path(report.fp32_onnx).with_suffix(".json").is_file()
    assert report.int8_onnx == ""
    assert payload["fp32_accepted"] is True
    assert json.loads((run / "eval.json").read_text(encoding="utf-8"))["split"] == "test"
