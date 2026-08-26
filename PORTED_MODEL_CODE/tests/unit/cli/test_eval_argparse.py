"""Argparse tests for smoke_detection.cli.eval."""

from __future__ import annotations

import pytest

from smoke_detection.cli import eval as eval_cli


def test_help_exits_zero():
    with pytest.raises(SystemExit) as exc:
        eval_cli.main(["--help"])
    assert exc.value.code == 0


def test_missing_config_exits_nonzero():
    with pytest.raises(SystemExit):
        eval_cli.main(["--ckpt", "nonexistent.ckpt"])


def test_missing_ckpt_exits_nonzero(classification_yaml_tmp):
    with pytest.raises(SystemExit):
        eval_cli.main(["--config", str(classification_yaml_tmp)])


def test_out_dir_defaults_to_output_plus_experiment(monkeypatch, classification_yaml_tmp, tmp_path):
    captured = {}

    def spy_eval_cls(cfg, ckpt, out_dir):
        captured["out_dir"] = out_dir

    monkeypatch.setattr(eval_cli, "_eval_classification", spy_eval_cls)

    fake_ckpt = tmp_path / "last.ckpt"
    fake_ckpt.write_bytes(b"")
    eval_cli.main(["--config", str(classification_yaml_tmp), "--ckpt", str(fake_ckpt)])

    # Path is cfg.paths.output_dir / cfg.paths.experiment_name / "eval"
    assert captured["out_dir"].name == "eval"
    assert captured["out_dir"].parent.name == "test_classification"


def test_out_dir_override_respected(monkeypatch, classification_yaml_tmp, tmp_path):
    captured = {}

    def spy_eval_cls(cfg, ckpt, out_dir):
        captured["out_dir"] = out_dir

    monkeypatch.setattr(eval_cli, "_eval_classification", spy_eval_cls)

    fake_ckpt = tmp_path / "last.ckpt"
    fake_ckpt.write_bytes(b"")
    custom = tmp_path / "custom_out"
    eval_cli.main(
        [
            "--config",
            str(classification_yaml_tmp),
            "--ckpt",
            str(fake_ckpt),
            "--out-dir",
            str(custom),
        ]
    )
    assert captured["out_dir"] == custom
