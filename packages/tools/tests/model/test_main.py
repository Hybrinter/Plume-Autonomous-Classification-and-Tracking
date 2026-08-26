"""Tests for python -m tools.model CLI dispatch."""

from pathlib import Path

import pytest
from tools.model.__main__ import main


def test_cli_unknown_exits() -> None:
    """An unknown subcommand raises SystemExit from argparse."""
    try:
        main(["nope"])
    except SystemExit as exc:
        assert exc.code != 0
    else:
        raise AssertionError("expected SystemExit")


def test_cli_train_without_torch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Train CLI returns 1 when torch is missing."""
    import tools.model.train as train_mod

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
