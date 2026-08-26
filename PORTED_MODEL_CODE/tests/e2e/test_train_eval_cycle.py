"""E2E: tiny train → eval cycle producing plots."""

from __future__ import annotations

from pathlib import Path

import pytest

from smoke_detection.cli.eval import main as eval_main
from smoke_detection.cli.train import main as train_main

pytestmark = pytest.mark.e2e


def _find_last_ckpt(output_dir: Path, experiment: str) -> Path:
    ckpt = output_dir / experiment
    matches = list(ckpt.rglob("last.ckpt"))
    assert matches, f"no last.ckpt under {ckpt}"
    return matches[0]


def test_classification_train_eval_cycle(
    classification_yaml_tmp, sample_classification_config, tmp_path
):
    cfg = sample_classification_config
    rc = train_main(
        [
            "--config",
            str(classification_yaml_tmp),
            "--override",
            "trainer.fast_dev_run=false",
            "--override",
            "trainer.max_epochs=1",
        ]
    )
    assert rc == 0

    ckpt = _find_last_ckpt(cfg.paths.output_dir, cfg.paths.experiment_name)
    rc = eval_main(["--config", str(classification_yaml_tmp), "--ckpt", str(ckpt)])
    assert rc == 0

    out_dir = cfg.paths.output_dir / cfg.paths.experiment_name / "eval"
    assert (out_dir / "confusion_matrix.png").stat().st_size > 0
    assert (out_dir / "roc_curve.png").stat().st_size > 0


def test_segmentation_train_eval_cycle(segmentation_yaml_tmp, sample_segmentation_config, tmp_path):
    cfg = sample_segmentation_config
    rc = train_main(
        [
            "--config",
            str(segmentation_yaml_tmp),
            "--override",
            "trainer.fast_dev_run=false",
            "--override",
            "trainer.max_epochs=1",
        ]
    )
    assert rc == 0

    ckpt = _find_last_ckpt(cfg.paths.output_dir, cfg.paths.experiment_name)
    rc = eval_main(["--config", str(segmentation_yaml_tmp), "--ckpt", str(ckpt)])
    assert rc == 0

    out_dir = cfg.paths.output_dir / cfg.paths.experiment_name / "eval"
    assert (out_dir / "iou_distribution.png").stat().st_size > 0
    assert (out_dir / "area_ratio_distribution.png").stat().st_size > 0
