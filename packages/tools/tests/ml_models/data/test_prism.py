"""Prism mix and 76 px proxy chips. No archives and no network."""

from __future__ import annotations

import importlib.util
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pytest
from tools.ml_models.data.bands import ZENODO_BAND_IDS
from tools.ml_models.data.pack import load_processed_pack
from tools.ml_models.data.prism import (
    PROXY_SIDE_PX,
    WeightTable,
    load_weight_table,
    mix_prism,
    to_proxy_chip,
    write_prism_pack,
)
from tools.ml_models.data.zenodo import TileIndex, TileRef

_ROOT = Path(__file__).resolve().parents[5]
_WEIGHTS = _ROOT / "data" / "manifests" / "ap3200t_s2_weights.toml"
_POLYGON = (np.array([[20.0, 20.0], [80.0, 20.0], [80.0, 80.0], [20.0, 80.0]], dtype=np.float32),)


def _stack_at(band_id: str, value: float, height: int, width: int) -> np.ndarray:
    """Return a 13-band stack that is ``value`` on one band and 0 elsewhere."""
    stack = np.zeros((len(ZENODO_BAND_IDS), height, width), dtype=np.float32)
    stack[ZENODO_BAND_IDS.index(band_id)] = value
    return stack


def test_mix_prism_b1_only_is_blue() -> None:
    """Counts of 10000 in B1 become blue 0.59 and leave green and red at 0."""
    table = load_weight_table(_WEIGHTS)
    assert table.id == "ap3200t_s2_figure"
    mixed = mix_prism(_stack_at("B1", 10000.0, 4, 5), ZENODO_BAND_IDS, table)
    assert mixed.shape == (3, 4, 5)
    assert mixed.dtype == np.float32
    assert mixed[0, 0, 0] == pytest.approx(0.59)
    assert mixed[1, 0, 0] == pytest.approx(0.0)
    assert mixed[2, 0, 0] == pytest.approx(0.0)
    assert np.allclose(mixed[0], 0.59)
    assert np.allclose(mixed[1], 0.0)
    assert np.allclose(mixed[2], 0.0)


def test_missing_weight_table_raises(tmp_path: Path) -> None:
    """A missing weight table is a FileNotFoundError."""
    missing = tmp_path / "absent.toml"
    with pytest.raises(FileNotFoundError):
        load_weight_table(missing)


def test_weight_sum_must_be_one(tmp_path: Path) -> None:
    """A color whose weights do not sum to 1 is refused."""
    path = tmp_path / "bad.toml"
    path.write_text(
        'id = "bad"\n\n[blue]\nB1 = 0.5\n\n[green]\nB3 = 1.0\n\n[red]\nB4 = 1.0\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="blue"):
        load_weight_table(path)


def test_unknown_band_id_raises() -> None:
    """A table band that is absent from the stack ids is refused."""
    table = WeightTable(id="x", blue={"NOPE": 1.0}, green={"B3": 1.0}, red={"B4": 1.0})
    stack = np.zeros((len(ZENODO_BAND_IDS), 2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="NOPE"):
        mix_prism(stack, ZENODO_BAND_IDS, table)


def test_to_proxy_chip_shape() -> None:
    """A 13 by 120 by 120 stack becomes a 76 px image and mask."""
    table = load_weight_table(_WEIGHTS)
    image, mask = to_proxy_chip(
        _stack_at("B1", 10000.0, 120, 120),
        ZENODO_BAND_IDS,
        table,
        _POLYGON,
    )
    assert image.shape == (3, PROXY_SIDE_PX, PROXY_SIDE_PX)
    assert mask.shape == (1, PROXY_SIDE_PX, PROXY_SIDE_PX)
    assert image.dtype == np.float32
    assert mask.dtype == np.float32
    assert image.shape[1] == 76
    assert np.allclose(image[0], 0.59)
    assert np.allclose(image[1], 0.0)
    assert np.allclose(image[2], 0.0)
    assert mask.max() == 1.0
    assert mask.min() == 0.0


def test_script_refuses_a_missing_weight_table(tmp_path: Path) -> None:
    """The CLI raises before it needs an image archive."""
    script = _ROOT / "scripts" / "preprocess_prism_proxy.py"
    spec = importlib.util.spec_from_file_location("preprocess_prism_proxy", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    missing = tmp_path / "no-weights.toml"
    with pytest.raises(FileNotFoundError, match="no-weights.toml"):
        module.main(
            [
                "--images",
                str(tmp_path / "images.tar"),
                "--labels",
                str(tmp_path / "labels.tar"),
                "--weights",
                str(missing),
                "--dest",
                str(tmp_path / "pack"),
            ]
        )
    assert not (tmp_path / "images.tar").exists()


def _tile(location: str, polygons: tuple[np.ndarray, ...] | None) -> TileRef:
    """Return one tile whose stem starts with ``location``."""
    stem = f"{location}_2019-01-01T00:00:00.000Z_0"
    return TileRef(
        stem=stem,
        location_id=location,
        positive=True,
        member_name=f"positive/{stem}.tif",
        polygons=polygons,
    )


def test_write_prism_pack_keeps_annotated_negatives(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty polygon tuple stays in the pack. A missing annotation does not."""
    polygon = (
        np.array([[20.0, 20.0], [80.0, 20.0], [80.0, 80.0], [20.0, 80.0]], dtype=np.float32),
    )
    tiles = (
        _tile("10", polygon),
        _tile("20", ()),
        _tile("30", polygon),
        _tile("40", polygon),
        _tile("50", None),
    )

    def _index(_images: Path, _labels: Path) -> TileIndex:
        return TileIndex(tiles=tiles)

    def _stacks(
        _images: Path,
        selected: Sequence[TileRef],
    ) -> Iterator[tuple[TileRef, np.ndarray, tuple[str, ...]]]:
        stack = np.zeros((len(ZENODO_BAND_IDS), 8, 8), dtype=np.float32)
        stack[0] = 10000.0
        descriptions = tuple("" for _band in ZENODO_BAND_IDS)
        for tile in selected:
            yield tile, stack, descriptions

    monkeypatch.setattr("tools.ml_models.data.prism.build_index", _index)
    monkeypatch.setattr("tools.ml_models.data.prism.iter_stacks", _stacks)
    dest = tmp_path / "pack"
    write_prism_pack(tmp_path / "images.tar", tmp_path / "labels.tar", _WEIGHTS, dest)
    pack = load_processed_pack(dest)
    assert pack.labels.shape == (4, 1)
    assert pack.labels.ravel().tolist() == [1.0, 0.0, 1.0, 1.0]
    assert float(pack.masks[1].max()) == 0.0
    assert float(pack.masks[0].max()) == 1.0
