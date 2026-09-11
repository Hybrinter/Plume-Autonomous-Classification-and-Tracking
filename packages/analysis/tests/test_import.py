"""Smoke test: the analysis package is importable."""

import analysis


def test_analysis_imports() -> None:
    """Importing analysis succeeds."""
    assert analysis.__name__ == "analysis"
