"""Tests for the AP-3200T weight table and color mix."""

from pathlib import Path

import numpy as np
import pytest
from tools.ml_models.dataset.sources.zenodo.bands import ZENODO_BAND_IDS
from tools.ml_models.dataset.sources.zenodo.prism import (
    OUTPUT_BANDS,
    WeightTable,
    load_weight_table,
    mix_prism,
)


def test_weight_table_loads_and_mixes(tmp_path: Path) -> None:
    """A simple table maps B2, B3, B4 straight through."""
    path = tmp_path / "w.toml"
    path.write_text(
        'id = "t1"\n[blue]\nB2 = 1.0\n[green]\nB3 = 1.0\n[red]\nB4 = 1.0\n',
        encoding="utf-8",
    )
    table = load_weight_table(path)
    assert table.id == "t1"
    assert table.color_map("blue") == {"B2": 1.0}
    stack = np.zeros((len(ZENODO_BAND_IDS), 4, 5), dtype=np.float32)
    stack[ZENODO_BAND_IDS.index("B2")] = 5000.0
    stack[ZENODO_BAND_IDS.index("B3")] = 2500.0
    stack[ZENODO_BAND_IDS.index("B4")] = 10000.0
    mixed = mix_prism(stack, ZENODO_BAND_IDS, table)
    assert OUTPUT_BANDS == ("BLUE", "GREEN", "RED")
    assert mixed.shape == (len(OUTPUT_BANDS), 4, 5)
    np.testing.assert_allclose(mixed[0], 0.5, atol=1e-6)
    np.testing.assert_allclose(mixed[1], 0.25, atol=1e-6)
    np.testing.assert_allclose(mixed[2], 1.0, atol=1e-6)


def test_weight_table_validation(tmp_path: Path) -> None:
    """Missing files, extra keys, and bad sums are rejected."""
    with pytest.raises(FileNotFoundError):
        load_weight_table(tmp_path / "absent.toml")
    bad = tmp_path / "bad.toml"
    bad.write_text(
        'id = "t1"\n[blue]\nB2 = 1.0\n[green]\nB3 = 1.0\n[red]\nB4 = 0.5\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sum"):
        load_weight_table(bad)
    extra = tmp_path / "extra.toml"
    extra.write_text(
        'id = "t1"\nnope = 1\n[blue]\nB2 = 1.0\n[green]\nB3 = 1.0\n[red]\nB4 = 1.0\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="keys mismatch"):
        load_weight_table(extra)


def test_mix_prism_rejects_unknown_table_band() -> None:
    """A weight on a band absent from the stack is rejected."""
    table = WeightTable(id="t", blue={"B13": 1.0}, green={"B3": 1.0}, red={"B4": 1.0})
    stack = np.zeros((len(ZENODO_BAND_IDS), 2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="unknown band id"):
        mix_prism(stack, ZENODO_BAND_IDS, table)
    with pytest.raises(ValueError, match="channels"):
        mix_prism(stack[:4], ZENODO_BAND_IDS, table)
    with pytest.raises(ValueError, match="unknown color"):
        table.color_map("nir")
