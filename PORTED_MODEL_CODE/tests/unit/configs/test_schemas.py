"""Tests for pydantic config schemas (base, classification, segmentation)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from smoke_detection.configs.base import OptimConfig, PathsConfig, TrainerConfig
from smoke_detection.configs.classification import ClassificationConfig
from smoke_detection.configs.segmentation import SegmentationConfig

# ----------------------------- TrainerConfig -----------------------------


def test_trainer_defaults():
    t = TrainerConfig()
    assert t.max_epochs == 50
    assert t.accelerator == "auto"
    assert t.precision == "32"
    assert t.fast_dev_run is False


def test_trainer_rejects_unknown_field():
    with pytest.raises(ValidationError):
        TrainerConfig(unknown_field=1)


def test_trainer_rejects_bad_precision():
    with pytest.raises(ValidationError):
        TrainerConfig(precision="64")


def test_trainer_rejects_bad_accelerator():
    with pytest.raises(ValidationError):
        TrainerConfig(accelerator="tpu")


# ----------------------------- PathsConfig -----------------------------


def test_paths_requires_experiment_name():
    with pytest.raises(ValidationError):
        PathsConfig()


def test_paths_accepts_strings_as_path():
    p = PathsConfig(data_root="data", output_dir="lightning_logs", experiment_name="e")
    assert isinstance(p.data_root, Path)


# ----------------------------- OptimConfig -----------------------------


def test_optim_requires_lr():
    with pytest.raises(ValidationError):
        OptimConfig()


def test_optim_scheduler_literal():
    with pytest.raises(ValidationError):
        OptimConfig(lr=1e-4, scheduler="exponential")


# ----------------------------- ClassificationConfig -----------------------------


def _minimal_classification_payload():
    return {
        "task": "classification",
        "trainer": {},
        "paths": {"experiment_name": "x"},
        "optim": {"lr": 1e-4},
        "model": {},
        "data": {},
    }


def test_classification_minimal_payload_validates():
    ClassificationConfig.model_validate(_minimal_classification_payload())


def test_classification_wrong_task_tag_rejected():
    payload = _minimal_classification_payload()
    payload["task"] = "segmentation"
    with pytest.raises(ValidationError):
        ClassificationConfig.model_validate(payload)


# ----------------------------- SegmentationConfig -----------------------------


def _minimal_segmentation_payload():
    return {
        "task": "segmentation",
        "trainer": {},
        "paths": {"experiment_name": "x"},
        "optim": {"lr": 1e-4},
        "model": {},
        "data": {},
    }


def test_segmentation_minimal_payload_validates():
    SegmentationConfig.model_validate(_minimal_segmentation_payload())


def test_cross_task_load_fails():
    """A classification YAML loaded through SegmentationConfig must be rejected."""
    payload = _minimal_classification_payload()
    with pytest.raises(ValidationError):
        SegmentationConfig.model_validate(payload)
