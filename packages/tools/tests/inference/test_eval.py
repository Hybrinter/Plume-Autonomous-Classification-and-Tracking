"""Eval tests."""

import json
from pathlib import Path

from tools.inference.eval import evaluate
from tools.inference.train import TrainConfig, train


def test_evaluate_writes_eval_json(tmp_path: Path) -> None:
    """evaluate scores the test split of a tiny synthetic run."""
    root = train(
        TrainConfig(
            kind="segmentor",
            epochs=1,
            batch_size=2,
            synthetic_samples=6,
            input_height_px=32,
            input_width_px=32,
            run_dir=str(tmp_path / "runs"),
            run_id="eval-me",
            seed=0,
        )
    )
    path = evaluate(root, split="test")
    assert path.is_file()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["split"] == "test"
    assert payload["n"] >= 1
    assert "mean_iou" in payload
    assert (root / "predictions.npz").is_file()
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert "test_mean_iou" in summary


def test_evaluate_defaults_to_val(tmp_path: Path) -> None:
    """evaluate without split writes val metrics into summary.json."""
    root = train(
        TrainConfig(
            kind="segmentor",
            epochs=1,
            batch_size=2,
            synthetic_samples=6,
            input_height_px=32,
            input_width_px=32,
            run_dir=str(tmp_path / "runs"),
            run_id="eval-val",
            seed=0,
        )
    )
    path = evaluate(root)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["split"] == "val"
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert "val_mean_iou" in summary
    assert summary["eval_split"] == "val"


def test_evaluate_test_split_keeps_val_keys(tmp_path: Path) -> None:
    """A later test eval does not drop val_* summary keys."""
    root = train(
        TrainConfig(
            kind="segmentor",
            epochs=1,
            batch_size=2,
            synthetic_samples=6,
            input_height_px=32,
            input_width_px=32,
            run_dir=str(tmp_path / "runs"),
            run_id="eval-both",
            seed=0,
        )
    )
    evaluate(root, split="val")
    evaluate(root, split="test")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert "val_mean_iou" in summary
    assert "test_mean_iou" in summary
