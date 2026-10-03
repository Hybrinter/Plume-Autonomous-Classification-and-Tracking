"""Tests for the flight tile directory."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from flight.libs.config import InferenceConfig
from flight.libs.types import Ok
from flight.payload.preprocess.band_select import select_bands
from flight.payload.preprocess.normalize import normalize_dn
from flight.payload.preprocess.tiling import slice_frame
from tools.ml_models.dataset.augment import AugmentRecipe
from tools.ml_models.dataset.build import build_dataset, build_flight
from tools.ml_models.dataset.gsd import to_model_gsd
from tools.ml_models.dataset.loader import ShardDataset
from tools.ml_models.dataset.raw import GsdPair
from tools.ml_models.dataset.sources.flight import (
    SCHEMA_VERSION,
    FlightTileDir,
    FlightTileWrite,
    write_flight_tile_dir,
)
from tools.ml_models.dataset.spec import BuildSpec
from tools.ml_models.dataset.store import read_gsd, read_images, read_labels, read_rows

_DEFAULT = InferenceConfig()
_TILE_HW = (
    _DEFAULT.input_height_px // _DEFAULT.tile_rows,
    _DEFAULT.input_width_px // _DEFAULT.tile_cols,
)


def _image(value: float) -> np.ndarray:
    """Return a constant float32 unit flight tile."""
    height, width = _TILE_HW
    return np.full((3, height, width), value, dtype=np.float32)


def test_layout_round_trip(tmp_path: Path) -> None:
    """Writer output reads back with the same pixels, mask, and group."""
    height, width = _TILE_HW
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
            image=_image(0.1),
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
            image=_image(0.2),
            mask=None,
            group_id=None,
        ),
    ]
    dest = tmp_path / "flight"
    write_flight_tile_dir(dest, tiles, source_ref="flight-test")
    source = FlightTileDir(dest)
    assert source.domain == "unit"
    assert source.tile_hw == _TILE_HW
    assert source.grid == (_DEFAULT.tile_rows, _DEFAULT.tile_cols)
    refs = source.index()
    assert refs[0].group_id == "group-a"
    assert refs[0].grid_rc == (0, 1)
    assert refs[0].height == height
    assert refs[0].width == width
    assert refs[1].group_id == "frame-b"
    loaded = list(source.iter_tiles())
    np.testing.assert_array_equal(loaded[0].image, _image(0.1))
    assert loaded[0].mask is not None
    assert loaded[0].mask[0, 0, 0] == 1
    assert loaded[1].mask is None
    assert not (dest / "masks" / "t1.npy").is_file()


def test_header_records_schema2_unit_layout(tmp_path: Path) -> None:
    """source.json records schema 2, unit domain, and float32 images."""
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
                image=_image(0.5),
            )
        ],
    )
    payload = json.loads((dest / "source.json").read_text(encoding="utf-8"))
    assert payload["schema"] == SCHEMA_VERSION
    assert payload["domain"] == "unit"
    assert payload["image_dtype"] == "float32"
    assert payload["tile_hw"] == list(_TILE_HW)
    assert payload["grid"] == [_DEFAULT.tile_rows, _DEFAULT.tile_cols]
    assert payload["band_names"] == list(_DEFAULT.input_bands)
    assert "bit_depth" not in payload


def test_old_header_is_rejected(tmp_path: Path) -> None:
    """A pre-schema-2 source.json fails with a rewrite message."""
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
                image=_image(0.5),
            )
        ],
    )
    path = dest / "source.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    old = {
        "band_names": payload["band_names"],
        "bit_depth": 12,
        "source_ref": payload["source_ref"],
        "gsd_reference_m": payload["gsd_reference_m"],
    }
    path.write_text(json.dumps(old) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        FlightTileDir(dest)


def test_stored_image_outside_unit_range_is_rejected_on_read(tmp_path: Path) -> None:
    """A tampered tile array fails the reader's finite unit-domain check."""
    dest = tmp_path / "flight"
    write_flight_tile_dir(dest, [_single_tile()])
    path = dest / "tiles" / "t0.npy"
    image = np.load(path)
    for value in (np.nan, np.inf, -0.25, 1.5):
        bad = image.copy()
        bad[0, 0, 0] = np.float32(value)
        np.save(path, bad)
        with pytest.raises(ValueError, match="finite inside"):
            list(FlightTileDir(dest).iter_tiles())
    np.save(path, image)


def test_recorded_layout_drives_reader_and_generic_build(tmp_path: Path) -> None:
    """A nondefault header, not InferenceConfig, sets bands and tile size."""
    images = [np.full((1, 4, 7), 0.25 * (index + 1), dtype=np.float32) for index in range(3)]
    tiles = [
        FlightTileWrite(
            tile_id=f"t{index}",
            frame_id=f"frame-{index}",
            row=index,
            col=index % 2,
            label=float(index % 2),
            theta_g_deg=15.0,
            gsd=GsdPair(15.87, 15.87),
            image=images[index],
        )
        for index in range(3)
    ]
    dest = tmp_path / "flight"
    write_flight_tile_dir(dest, tiles, band_names=("BLUE",), tile_hw=(4, 7), grid=(3, 2))
    source = FlightTileDir(dest)
    assert source.band_names == ("BLUE",)
    assert source.tile_hw == (4, 7)
    assert source.grid == (3, 2)
    refs = source.index()
    assert [(ref.height, ref.width) for ref in refs] == [(4, 7)] * 3
    assert [ref.grid_rc for ref in refs] == [(0, 0), (1, 1), (2, 0)]
    for tile in source.iter_tiles():
        assert tile.image.shape == (1, 4, 7)
    dataset_dir = tmp_path / "ds"
    manifest = build_dataset(source, dataset_dir, BuildSpec())
    assert manifest.band_names == ("BLUE",)
    shard_dirs = [
        shard
        for split_name in ("train", "val", "test")
        if (dataset_dir / "classifier" / split_name).is_dir()
        for shard in (dataset_dir / "classifier" / split_name).iterdir()
    ]
    assert {shard.name for shard in shard_dirs} == {"4x7"}
    found = False
    for shard in shard_dirs:
        rows = read_rows(shard)
        stored = read_images(shard)
        assert stored.shape[1:] == (1, 4, 7)
        for index, row in enumerate(rows):
            if row.tile_id == "t0" and row.element == "id":
                found = True
                np.testing.assert_array_equal(stored[index], images[0])
    assert found


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
                image=_image(0.1),
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
        image=_image(0.1),
    )
    with pytest.raises(ValueError, match="invalid flight tile capture"):
        write_flight_tile_dir(tmp_path / "flight", [tile])
    with pytest.raises(ValueError, match="invalid flight tile capture"):
        write_flight_tile_dir(tmp_path / "flight2", [replace(tile, tile_id="..")])


def test_grid_indices_outside_grid_are_rejected_on_write(tmp_path: Path) -> None:
    """row and col must lie inside the recorded grid on write."""
    tile = FlightTileWrite(
        tile_id="t0",
        frame_id="frame-a",
        row=8,
        col=0,
        label=1.0,
        theta_g_deg=15.0,
        gsd=GsdPair(15.87, 15.87),
        image=_image(0.1),
    )
    with pytest.raises(ValueError, match="invalid flight tile capture"):
        write_flight_tile_dir(tmp_path / "flight", [tile])
    with pytest.raises(ValueError, match="invalid flight tile capture"):
        write_flight_tile_dir(tmp_path / "flight2", [replace(tile, row=0, col=-1)])


def test_non_unit_image_is_rejected_on_write(tmp_path: Path) -> None:
    """uint16, float64, and out-of-range float32 images cannot be written."""
    base = FlightTileWrite(
        tile_id="t0",
        frame_id="frame-a",
        row=0,
        col=0,
        label=1.0,
        theta_g_deg=15.0,
        gsd=GsdPair(15.87, 15.87),
        image=_image(0.5),
    )
    height, width = _TILE_HW
    bad_images: tuple[np.ndarray, ...] = (
        np.zeros((3, height, width), dtype=np.uint16),
        np.full((3, height, width), 0.5, dtype=np.float64),
        np.full((3, height, width), 1.5, dtype=np.float32),
    )
    for index, image in enumerate(bad_images):
        with pytest.raises(ValueError):
            write_flight_tile_dir(tmp_path / f"flight{index}", [replace(base, image=image)])


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
                image=_image(0.1),
            )
        ],
    )
    path = dest / "index.jsonl"
    row = json.loads(path.read_text(encoding="utf-8"))
    row["tile_id"] = "../t0"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="valid unit tile capture"):
        FlightTileDir(dest)
    row["tile_id"] = "t0"
    row["col"] = 8
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="valid unit tile capture"):
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
        "image": _image(0.1),
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


def test_normalized_frame_tile_round_trips_through_build(tmp_path: Path) -> None:
    """A normalize/select/slice tile survives write, import, and build."""
    frame = np.zeros((3, 8, 14), dtype=np.float32)
    frame[:, 4, 7] = np.float32(0.0)
    frame[:, 4, 8] = np.float32(4095.0)
    frame[2, 6, 10] = np.float32(4095.0 * 0.1234567)
    normalized = normalize_dn(frame, 12)
    selected = select_bands(normalized, ("RED", "GREEN", "BLUE"), ("BLUE", "GREEN", "RED"))
    assert isinstance(selected, Ok)
    np.testing.assert_array_equal(selected.value[0], normalized[2])
    sliced = slice_frame(selected.value[None], (2, 2))
    assert isinstance(sliced, Ok)
    tile_image = np.ascontiguousarray(sliced.value[3])
    assert tile_image.shape == (3, 4, 7)
    assert float(tile_image.min()) == 0.0
    assert float(tile_image.max()) == 1.0
    assert tile_image[0, 2, 3] == normalized[2, 6, 10] != 0.0

    source_dir = tmp_path / "flight-src"
    write_flight_tile_dir(
        source_dir,
        [
            FlightTileWrite(
                tile_id=f"tile{index}",
                frame_id=f"frame-{index}",
                row=1,
                col=1,
                label=1.0,
                theta_g_deg=15.0,
                gsd=GsdPair(16.5, 17.1),
                image=tile_image,
                group_id=f"group-{index}",
            )
            for index in range(3)
        ],
        band_names=("BLUE", "GREEN", "RED"),
        tile_hw=(4, 7),
        grid=(2, 2),
    )
    dataset_dir = tmp_path / "dataset"
    manifest = build_flight(
        source_dir,
        dataset_dir,
        BuildSpec(augment=AugmentRecipe(elements=("id",))),
    )
    assert manifest.band_names == ("BLUE", "GREEN", "RED")
    assert manifest.source == "flight"
    assert not (dataset_dir / "segmentor").exists()

    expected_gsd = np.array([16.5, 17.1], dtype=np.float32)
    expected_model_gsd = to_model_gsd(expected_gsd, manifest.gsd_reference_m)
    found = False
    for split in ("train", "val", "test"):
        shard_dir = dataset_dir / "classifier" / split / "4x7"
        if not shard_dir.is_dir():
            continue
        rows = read_rows(shard_dir)
        assert {row.element for row in rows} == {"id"}
        np.testing.assert_array_equal(
            read_gsd(shard_dir),
            np.broadcast_to(expected_gsd, (len(rows), 2)),
        )
        index = next(
            (i for i, row in enumerate(rows) if row.tile_id == "tile0" and row.element == "id"),
            None,
        )
        if index is None:
            continue
        found = True
        assert rows[index].group_id == "group-0"
        assert rows[index].frame_id == "frame-0"
        assert rows[index].grid_rc == (1, 1)
        np.testing.assert_array_equal(read_labels(shard_dir)[index], np.array([1.0], np.float32))
        dataset = ShardDataset(shard_dir, manifest.gsd_reference_m, "classifier", channels=3)
        image, encoded, target = dataset[index]
        np.testing.assert_array_equal(image.numpy(), tile_image)
        np.testing.assert_array_equal(encoded.numpy(), expected_model_gsd)
        np.testing.assert_array_equal(target.numpy(), np.array([1.0], dtype=np.float32))
    assert found


def test_huge_header_reference_is_a_value_error(tmp_path: Path) -> None:
    """A huge JSON integer gsd_reference_m fails cleanly, not OverflowError."""
    dest = tmp_path / "flight"
    write_flight_tile_dir(dest, [_single_tile()])
    path = dest / "source.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["gsd_reference_m"] = 10**400
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="layout"):
        FlightTileDir(dest)


def test_huge_index_numbers_are_value_errors(tmp_path: Path) -> None:
    """Huge JSON integers in GSD and theta fields fail cleanly."""
    dest = tmp_path / "flight"
    write_flight_tile_dir(dest, [_single_tile()])
    path = dest / "index.jsonl"
    original = json.loads(path.read_text(encoding="utf-8").strip())
    for key in ("theta_g_deg", "gsd_lateral_m"):
        payload = dict(original)
        payload[key] = 10**400
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match=key):
            FlightTileDir(dest)
