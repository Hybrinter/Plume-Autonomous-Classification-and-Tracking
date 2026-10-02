"""Tests for the flight tile directory."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from tools.ml_models.dataset.geometry import tile_hw
from tools.ml_models.dataset.raw import GsdPair
from tools.ml_models.dataset.sources.flight import (
    FlightTileDir,
    FlightTileWrite,
    write_flight_tile_dir,
)


def _image(value: int) -> np.ndarray:
    """Return a constant uint16 flight tile."""
    height, width = tile_hw()
    return np.full((3, height, width), value, dtype=np.uint16)


def test_layout_round_trip(tmp_path: Path) -> None:
    """Writer output reads back with the same pixels, mask, and group."""
    height, width = tile_hw()
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[0, 0] = 1
    tiles = [
        FlightTileWrite(
            tile_id="t0",
            frame_id="frame-a",
            row=0,
            col=1,
            label=1.0,
            theta_g_deg=15.0,
            gsd=GsdPair(16.5, 17.1),
            image=_image(10),
            mask=mask,
            group_id="group-a",
        ),
        FlightTileWrite(
            tile_id="t1",
            frame_id="frame-b",
            row=1,
            col=0,
            label=0.0,
            theta_g_deg=5.0,
            gsd=GsdPair(15.87, 15.87),
            image=_image(20),
            mask=None,
            group_id=None,
        ),
    ]
    dest = tmp_path / "flight"
    write_flight_tile_dir(dest, tiles, source_ref="flight-test")
    source = FlightTileDir(dest)
    refs = source.index()
    assert refs[0].group_id == "group-a"
    assert refs[0].grid_rc == (0, 1)
    assert refs[1].group_id == "frame-b"
    loaded = list(source.iter_tiles())
    np.testing.assert_array_equal(loaded[0].image, _image(10))
    assert loaded[0].mask is not None
    assert loaded[0].mask[0, 0, 0] == 1
    assert loaded[1].mask is None
    assert not (dest / "masks" / "t1.npy").is_file()


def test_missing_group_id_defaults_to_frame(tmp_path: Path) -> None:
    """An index row without group_id uses frame_id."""
    dest = tmp_path / "flight"
    write_flight_tile_dir(
        dest,
        [
            FlightTileWrite(
                tile_id="t0",
                frame_id="frame-z",
                row=0,
                col=0,
                label=1.0,
                theta_g_deg=0.0,
                gsd=GsdPair(15.87, 15.87),
                image=_image(1),
            )
        ],
    )
    path = dest / "index.jsonl"
    row = json.loads(path.read_text(encoding="utf-8"))
    del row["group_id"]
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert FlightTileDir(dest).index()[0].group_id == "frame-z"


def test_traversal_tile_id_is_rejected_on_write(tmp_path: Path) -> None:
    """A tile_id that is not a file stem cannot be written."""
    tile = FlightTileWrite(
        tile_id="../t0",
        frame_id="frame-a",
        row=0,
        col=0,
        label=1.0,
        theta_g_deg=15.0,
        gsd=GsdPair(15.87, 15.87),
        image=_image(10),
    )
    with pytest.raises(ValueError, match="file stem"):
        write_flight_tile_dir(tmp_path / "flight", [tile])
    with pytest.raises(ValueError, match="file stem"):
        write_flight_tile_dir(tmp_path / "flight2", [replace(tile, tile_id="..")])


def test_grid_indices_outside_grid_are_rejected_on_write(tmp_path: Path) -> None:
    """row and col must lie in 0..7 on write."""
    tile = FlightTileWrite(
        tile_id="t0",
        frame_id="frame-a",
        row=8,
        col=0,
        label=1.0,
        theta_g_deg=15.0,
        gsd=GsdPair(15.87, 15.87),
        image=_image(10),
    )
    with pytest.raises(ValueError, match="0..7"):
        write_flight_tile_dir(tmp_path / "flight", [tile])
    with pytest.raises(ValueError, match="0..7"):
        write_flight_tile_dir(tmp_path / "flight2", [replace(tile, row=0, col=-1)])


def test_index_rows_must_use_stems_and_grid_bounds(tmp_path: Path) -> None:
    """A bad tile_id or grid index in index.jsonl is rejected on read."""
    dest = tmp_path / "flight"
    write_flight_tile_dir(
        dest,
        [
            FlightTileWrite(
                tile_id="t0",
                frame_id="frame-a",
                row=0,
                col=0,
                label=1.0,
                theta_g_deg=15.0,
                gsd=GsdPair(15.87, 15.87),
                image=_image(10),
            )
        ],
    )
    path = dest / "index.jsonl"
    row = json.loads(path.read_text(encoding="utf-8"))
    row["tile_id"] = "../t0"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="file stem"):
        FlightTileDir(dest)
    row["tile_id"] = "t0"
    row["col"] = 8
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="0..7"):
        FlightTileDir(dest)


def _single_tile(theta_g_deg: float = 15.0, **overrides: object) -> FlightTileWrite:
    """One minimal valid flight tile with ``theta_g_deg`` degrees."""
    fields: dict[str, object] = {
        "tile_id": "t0",
        "frame_id": "frame-a",
        "row": 0,
        "col": 0,
        "label": 1.0,
        "theta_g_deg": theta_g_deg,
        "gsd": GsdPair(15.87, 15.87),
        "image": _image(1),
    }
    fields.update(overrides)
    return FlightTileWrite(**fields)  # type: ignore[arg-type]


def test_theta_maps_to_nearest_elevation_bin(tmp_path: Path) -> None:
    """bin_id is elevation{nearest} with ties choosing the smaller value."""
    cases = {
        -5.0: "elevation5",
        5.0: "elevation5",
        9.9: "elevation5",
        10.0: "elevation5",
        10.1: "elevation15",
        15.0: "elevation15",
        20.0: "elevation15",
        25.0: "elevation25",
        30.0: "elevation25",
        33.0: "elevation35",
        40.0: "elevation35",
        45.0: "elevation45",
        90.0: "elevation45",
    }
    tiles = [_single_tile(theta, tile_id=f"t{index}") for index, theta in enumerate(cases)]
    dest = tmp_path / "flight"
    write_flight_tile_dir(dest, tiles)
    refs = FlightTileDir(dest).index()
    for ref, expected in zip(refs, cases.values(), strict=True):
        assert ref.bin_id == expected
        assert ref.gsd_nominal is False


def test_gsd_nominal_round_trip(tmp_path: Path) -> None:
    """The writer emits gsd_nominal and the reader returns it strictly."""
    dest = tmp_path / "flight"
    write_flight_tile_dir(dest, [_single_tile(gsd_nominal=True)])
    payload = json.loads((dest / "index.jsonl").read_text(encoding="utf-8").strip())
    assert payload["gsd_nominal"] is True
    ref = FlightTileDir(dest).index()[0]
    assert ref.gsd_nominal is True
    assert ref.theta_g_deg == 15.0


def test_gsd_nominal_must_be_a_boolean(tmp_path: Path) -> None:
    """Non-boolean gsd_nominal values in index.jsonl are rejected."""
    dest = tmp_path / "flight"
    write_flight_tile_dir(dest, [_single_tile()])
    path = dest / "index.jsonl"
    original = json.loads(path.read_text(encoding="utf-8").strip())
    for bad in (1, "yes"):
        payload = dict(original)
        payload["gsd_nominal"] = bad
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="gsd_nominal"):
            FlightTileDir(dest)


def test_theta_required_and_finite(tmp_path: Path) -> None:
    """index.jsonl requires a finite numeric theta_g_deg."""
    dest = tmp_path / "flight"
    write_flight_tile_dir(dest, [_single_tile()])
    path = dest / "index.jsonl"
    original = json.loads(path.read_text(encoding="utf-8").strip())
    payload = dict(original)
    del payload["theta_g_deg"]
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        FlightTileDir(dest)
    for bad in (True, "15", float("nan")):
        payload = dict(original)
        payload["theta_g_deg"] = bad
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="theta_g_deg"):
            FlightTileDir(dest)
