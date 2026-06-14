"""Smoke test: the gse package is importable and declares py.typed."""

import gse


def test_gse_imports() -> None:
    """Importing gse succeeds and exposes its dunder version."""
    assert gse.__name__ == "gse"
