"""Tests for the root pact-tools command dispatcher."""

from tools.cli import main


def test_root_help_succeeds() -> None:
    """The root command exposes Typer help."""
    assert main(["--help"]) == 0


def test_root_registers_inference_commands() -> None:
    """The root command dispatches to the inference application."""
    assert main(["inference", "--help"]) == 0


def test_root_registers_analysis_commands() -> None:
    """The root command dispatches to the analysis application."""
    assert main(["analysis", "list"]) == 0
