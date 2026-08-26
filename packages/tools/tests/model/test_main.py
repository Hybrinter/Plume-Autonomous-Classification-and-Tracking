"""Tests for python -m tools.model CLI dispatch."""

from tools.model.__main__ import main


def test_cli_train_exits_not_implemented() -> None:
    """The train subcommand exits 2 in this scaffold layer."""
    assert main(["train"]) == 2


def test_cli_unknown_exits() -> None:
    """An unknown subcommand raises SystemExit from argparse."""
    try:
        main(["nope"])
    except SystemExit as exc:
        assert exc.code != 0
    else:
        raise AssertionError("expected SystemExit")
