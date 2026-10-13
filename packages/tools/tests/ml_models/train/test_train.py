"""Tests for the unavailable standard-training boundary."""

from collections.abc import Callable
from pathlib import Path

from flight.libs.types import Err
from tools.ml_models.train.config import TrainConfig
from tools.ml_models.train.loop import train


def test_train_returns_unavailable_before_any_output(
    tmp_path: Path, build_synthetic_dataset: Callable[..., Path]
) -> None:
    """A valid config over a finished dataset still fails closed."""
    dataset = build_synthetic_dataset(tmp_path / "ds", n=9)
    cfg = TrainConfig(
        kind="classifier",
        arch="pactnet_w8_d2",
        dataset=str(dataset),
        epochs=1,
        device="cpu",
        run_dir=str(tmp_path / "runs"),
        run_id="fixed",
    )
    result = train(cfg)
    assert isinstance(result, Err)
    assert "unavailable" in result.error
    assert not (tmp_path / "runs").exists()


def test_train_default_config_returns_unavailable() -> None:
    """``train()`` with defaults hits the same explicit error."""
    result = train()
    assert isinstance(result, Err)
    assert "unavailable" in result.error


def test_cli_train_fails_unavailable(tmp_path: Path) -> None:
    """``ml-models train`` exits nonzero and creates no run directory."""
    from tools.ml_models.cli import main

    code = main(
        [
            "train",
            "--kind",
            "classifier",
            "--arch",
            "pactnet_w8_d2",
            "--dataset",
            str(tmp_path / "ds"),
            "--run-dir",
            str(tmp_path / "runs"),
            "--epochs",
            "1",
            "--device",
            "cpu",
        ]
    )
    assert code != 0
    assert not (tmp_path / "runs").exists()


def test_cli_train_rejects_plural_dataset_and_missing_config(tmp_path: Path) -> None:
    """A repeated ``--dataset`` or a missing config still fails fast."""
    from tools.ml_models.cli import main

    dataset = str(tmp_path / "ds")
    assert main(["train", "--dataset", dataset, "--dataset", dataset]) != 0
    assert main(["train", "--config", str(tmp_path / "missing.toml")]) != 0
