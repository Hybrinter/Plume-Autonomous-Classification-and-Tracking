"""Tests for the tools.inference Typer CLI."""

from pathlib import Path

from tools.inference.cli import main


def test_cli_unknown_returns_nonzero() -> None:
    """An unknown subcommand returns a Click usage-error exit code."""
    assert main(["nope"]) != 0


def test_cli_train_unknown_arch(tmp_path: Path) -> None:
    """Train CLI returns 1 when the architecture name is unknown."""
    code = main(
        [
            "train",
            "--kind",
            "segmentor",
            "--arch",
            "nope",
            "--run-dir",
            str(tmp_path),
            "--epochs",
            "1",
            "--height",
            "32",
            "--width",
            "32",
        ]
    )
    assert code == 1
