"""Smoke test confirming the sim package and its subpackages import."""

import importlib


def test_sim_subpackages_import() -> None:
    """The sim package and its subpackages import without error."""
    for name in ("sim", "sim.sil", "sim.scene", "sim.twin"):
        assert importlib.import_module(name) is not None
