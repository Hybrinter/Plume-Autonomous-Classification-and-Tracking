"""Integration: full YAML-to-trainer wiring with fast_dev_run."""

from __future__ import annotations

from pathlib import Path

from smoke_detection.cli.train import (
    _build_classification,
    _build_segmentation,
    _build_trainer,
)


def _run(cfg):
    if cfg.task == "classification":
        module, dm, monitor = _build_classification(cfg)
    else:
        module, dm, monitor = _build_segmentation(cfg)
    trainer = _build_trainer(cfg, monitor=monitor)
    trainer.fit(module, datamodule=dm)
    return trainer


def test_classification_config_to_trainer_fit(sample_classification_config):
    trainer = _run(sample_classification_config)
    # fast_dev_run: no checkpoints are written, but training completed.
    assert trainer.state.finished


def test_segmentation_config_to_trainer_fit(sample_segmentation_config):
    trainer = _run(sample_segmentation_config)
    assert trainer.state.finished


def test_checkpoint_written_without_fast_dev_run(sample_classification_config):
    cfg = sample_classification_config.model_copy(deep=True)
    cfg.trainer.fast_dev_run = False
    cfg.trainer.max_epochs = 1
    _run(cfg)
    ckpt_dir = Path(cfg.paths.output_dir) / cfg.paths.experiment_name
    assert ckpt_dir.exists()
    ckpts = list(ckpt_dir.rglob("*.ckpt"))
    assert len(ckpts) >= 1
