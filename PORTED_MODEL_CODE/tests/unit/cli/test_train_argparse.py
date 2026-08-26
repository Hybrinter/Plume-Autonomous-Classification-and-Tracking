"""Argparse- and dispatch-level tests for smoke_detection.cli.train."""

from __future__ import annotations

import pytest

from smoke_detection.cli import train as train_cli


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        train_cli.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--config" in out


def test_missing_config_exits_nonzero():
    with pytest.raises(SystemExit):
        train_cli.main([])  # --config is required


def test_override_accumulates(monkeypatch, tmp_path, classification_yaml_tmp):
    """Calling main with two --override flags must pass both into load_config."""
    captured = {}

    from smoke_detection.configs import loader

    real_load = loader.load_config

    def spy(path, overrides=None):
        captured["overrides"] = list(overrides or [])
        return real_load(path, overrides=overrides)

    monkeypatch.setattr("smoke_detection.cli.train.load_config", spy)

    # Prevent actual training
    import lightning as L

    monkeypatch.setattr(L.Trainer, "fit", lambda self, *a, **k: None)

    train_cli.main(
        [
            "--config",
            str(classification_yaml_tmp),
            "--override",
            "optim.lr=5e-4",
            "--override",
            "trainer.max_epochs=2",
        ]
    )
    assert captured["overrides"] == ["optim.lr=5e-4", "trainer.max_epochs=2"]


def test_classification_dispatch_calls_classification_builder(monkeypatch, classification_yaml_tmp):
    """A classification YAML must route through _build_classification."""
    called = {}

    real = train_cli._build_classification

    def spy(cfg):
        called["yes"] = True
        return real(cfg)

    monkeypatch.setattr(train_cli, "_build_classification", spy)

    import lightning as L

    monkeypatch.setattr(L.Trainer, "fit", lambda self, *a, **k: None)

    train_cli.main(["--config", str(classification_yaml_tmp)])
    assert called.get("yes") is True


def test_segmentation_dispatch_calls_segmentation_builder(monkeypatch, segmentation_yaml_tmp):
    called = {}

    real = train_cli._build_segmentation

    def spy(cfg):
        called["yes"] = True
        return real(cfg)

    monkeypatch.setattr(train_cli, "_build_segmentation", spy)

    import lightning as L

    monkeypatch.setattr(L.Trainer, "fit", lambda self, *a, **k: None)

    train_cli.main(["--config", str(segmentation_yaml_tmp)])
    assert called.get("yes") is True
