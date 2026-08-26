"""Tests for YAML config loading + dotted overrides."""

from __future__ import annotations

import pytest

from smoke_detection.configs.classification import ClassificationConfig
from smoke_detection.configs.loader import _apply_dotted_override, _coerce_scalar, load_config
from smoke_detection.configs.segmentation import SegmentationConfig


def _write_yaml(path, body):
    path.write_text(body)
    return path


def test_load_classification_yaml(tmp_path):
    cfg = load_config("configs/classification/default.yaml")
    assert isinstance(cfg, ClassificationConfig)
    assert cfg.task == "classification"


def test_load_segmentation_yaml(tmp_path):
    cfg = load_config("configs/segmentation/default.yaml")
    assert isinstance(cfg, SegmentationConfig)
    assert cfg.task == "segmentation"


def test_dotted_override_scalar(tmp_path):
    p = _write_yaml(
        tmp_path / "c.yaml",
        "task: classification\n"
        "trainer: {}\n"
        "paths: {experiment_name: x}\n"
        "optim: {lr: 1e-4}\n"
        "model: {}\n"
        "data: {}\n",
    )
    cfg = load_config(p, overrides=["optim.lr=5e-3"])
    assert cfg.optim.lr == 5e-3


def test_dotted_override_creates_missing_dicts(tmp_path):
    p = _write_yaml(
        tmp_path / "c.yaml",
        "task: classification\n"
        "trainer: {}\n"
        "paths: {experiment_name: x}\n"
        "optim: {lr: 1e-4}\n"
        "model: {}\n"
        "data: {}\n",
    )
    cfg = load_config(p, overrides=["trainer.fast_dev_run=true"])
    assert cfg.trainer.fast_dev_run is True


def test_override_syntax_error_raises(tmp_path):
    p = _write_yaml(
        tmp_path / "c.yaml",
        "task: classification\ntrainer: {}\npaths: {experiment_name: x}\noptim: {lr: 1e-4}\nmodel: {}\ndata: {}\n",
    )
    with pytest.raises(ValueError, match="key=value"):
        load_config(p, overrides=["just-a-key"])


def test_override_descent_into_scalar_raises():
    raw = {"a": 1}
    with pytest.raises(ValueError, match="non-mapping"):
        _apply_dotted_override(raw, "a.b=2")


def test_missing_task_raises(tmp_path):
    p = _write_yaml(
        tmp_path / "c.yaml", "trainer: {}\npaths: {experiment_name: x}\noptim: {lr: 1e-4}\n"
    )
    with pytest.raises(ValueError, match="task"):
        load_config(p)


def test_invalid_task_raises(tmp_path):
    p = _write_yaml(
        tmp_path / "c.yaml",
        "task: bogus\ntrainer: {}\npaths: {experiment_name: x}\noptim: {lr: 1e-4}\n",
    )
    with pytest.raises(ValueError, match="task"):
        load_config(p)


@pytest.mark.parametrize(
    "value,expected",
    [
        ("true", True),
        ("false", False),
        ("True", True),
        ("FALSE", False),
        ("null", None),
        ("None", None),
        ("none", None),
        ("42", 42),
        ("-3", -3),
        ("1.5", 1.5),
        ("1e-4", 1e-4),
        ("2.5e3", 2500.0),
        ("hello", "hello"),
        ("", ""),
    ],
)
def test_coerce_scalar(value, expected):
    assert _coerce_scalar(value) == expected
