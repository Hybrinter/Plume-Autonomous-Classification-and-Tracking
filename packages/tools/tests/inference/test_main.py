"""Tests for the tools.inference Typer CLI."""

from pathlib import Path

import pytest
from tools.inference.cli import main


def test_cli_unknown_returns_nonzero() -> None:
    """An unknown subcommand returns a Click usage-error exit code."""
    assert main(["nope"]) != 0


def test_cli_train_without_torch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Train CLI returns 1 when torch is missing."""
    import tools.inference.train as train_mod

    def _boom() -> object:
        raise ImportError("torch is required")

    monkeypatch.setattr(train_mod, "_import_torch", _boom)
    code = main(
        [
            "train",
            "--kind",
            "segmentor",
            "--out",
            str(tmp_path / "x.pt"),
            "--epochs",
            "1",
            "--height",
            "32",
            "--width",
            "32",
        ]
    )
    assert code == 1
