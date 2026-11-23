"""Tests for rows.jsonl decode and the schema-1 field defaults."""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from tools.ml_models.dataset.store import RowRecord, ShardWriter, read_images, read_rows


def _write_one_row(
    shard_dir: Path,
    row: RowRecord,
    *,
    with_masks: bool = False,
) -> Path:
    """Write a single-row shard and return its ``rows.jsonl`` path."""
    writer = ShardWriter(shard_dir, 1, 4, 8, channels=3, with_masks=with_masks)
    image = np.zeros((3, 4, 8), dtype=np.float32)
    mask = np.zeros((1, 4, 8), dtype=np.uint8) if with_masks else None
    writer.append(image, np.array([10.0, 20.0], dtype=np.float32), 1.0, mask, row)
    writer.close()
    return shard_dir / "rows.jsonl"


def test_writer_allocates_explicit_channel_count(tmp_path: Path) -> None:
    """A non-three-channel shard stores float32 (N, C, H, W) images."""
    writer = ShardWriter(tmp_path / "shard", 1, 4, 8, channels=1, with_masks=False)
    row = RowRecord(
        tile_id="t0",
        group_id="g0",
        frame_id="f0",
        grid_rc=(0, 0),
        bin_id="",
        element="id",
    )
    image = np.full((1, 4, 8), 0.5, dtype=np.float32)
    writer.append(image, np.array([10.0, 20.0], dtype=np.float32), 1.0, None, row)
    writer.close()
    stored = read_images(tmp_path / "shard")
    assert stored.dtype == np.float32
    np.testing.assert_array_equal(stored[0], image)


def test_writer_rejects_wrong_image_dtype(tmp_path: Path) -> None:
    """A uint16 image cannot enter a float32 shard."""
    writer = ShardWriter(tmp_path / "shard", 1, 4, 8, channels=3, with_masks=False)
    row = RowRecord(
        tile_id="t0",
        group_id="g0",
        frame_id="f0",
        grid_rc=(0, 0),
        bin_id="",
        element="id",
    )
    with pytest.raises(ValueError, match="float32"):
        writer.append(
            np.zeros((3, 4, 8), dtype=np.uint16),
            np.array([10.0, 20.0], dtype=np.float32),
            1.0,
            None,
            row,
        )
    writer.abort()


def test_writer_always_emits_angle_and_nominal(tmp_path: Path) -> None:
    """Rows always carry theta_g_deg and gsd_nominal, even at defaults."""
    row = RowRecord(
        tile_id="t0",
        group_id="g0",
        frame_id="f0",
        grid_rc=(0, 0),
        bin_id="elevation15",
        element="id",
    )
    path = _write_one_row(tmp_path / "shard", row)
    payload = json.loads(path.read_text(encoding="utf-8").strip())
    assert payload["theta_g_deg"] is None
    assert payload["gsd_nominal"] is False
    loaded = read_rows(tmp_path / "shard")
    assert loaded[0] == row


def test_schema1_rows_decode_with_defaults(tmp_path: Path) -> None:
    """A row written before the angle and nominal fields decodes cleanly."""
    row = RowRecord(
        tile_id="t0",
        group_id="g0",
        frame_id="f0",
        grid_rc=(1, 2),
        bin_id="elevation25",
        element="id",
        theta_g_deg=25.0,
        gsd_nominal=True,
    )
    path = _write_one_row(tmp_path / "shard", row)
    payload = json.loads(path.read_text(encoding="utf-8").strip())
    del payload["theta_g_deg"]
    del payload["gsd_nominal"]
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    loaded = read_rows(tmp_path / "shard")
    assert loaded[0].theta_g_deg is None
    assert loaded[0].gsd_nominal is False
    assert loaded[0].tile_id == "t0"
    assert loaded[0].bin_id == "elevation25"


def test_new_fields_round_trip(tmp_path: Path) -> None:
    """A populated theta_g_deg and gsd_nominal survive the codec."""
    row = RowRecord(
        tile_id="t0",
        group_id="g0",
        frame_id="f0",
        grid_rc=(3, 4),
        bin_id="elevation45",
        element="id",
        theta_g_deg=44.5,
        gsd_nominal=True,
    )
    _write_one_row(tmp_path / "shard", row)
    loaded = read_rows(tmp_path / "shard")
    assert loaded[0] == row
    assert loaded[0].theta_g_deg == 44.5
    assert loaded[0].gsd_nominal is True


def test_new_fields_are_strictly_typed(tmp_path: Path) -> None:
    """gsd_nominal must be a JSON bool and theta_g_deg a finite number."""
    row = RowRecord(
        tile_id="t0",
        group_id="g0",
        frame_id="f0",
        grid_rc=(0, 0),
        bin_id="",
        element="id",
    )
    path = _write_one_row(tmp_path / "shard", row)
    original = json.loads(path.read_text(encoding="utf-8").strip())
    for bad in (1, "yes", None):
        payload = dict(original)
        payload["gsd_nominal"] = bad
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="gsd_nominal"):
            read_rows(tmp_path / "shard")
    for bad_theta in (True, "5", float("nan"), float("inf")):
        payload = dict(original)
        payload["theta_g_deg"] = bad_theta
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="theta_g_deg"):
            read_rows(tmp_path / "shard")


def test_unknown_and_missing_keys_rejected(tmp_path: Path) -> None:
    """Extra keys and absent original keys are still rejected."""
    row = RowRecord(
        tile_id="t0",
        group_id="g0",
        frame_id="f0",
        grid_rc=(0, 0),
        bin_id="",
        element="id",
    )
    path = _write_one_row(tmp_path / "shard", row)
    original = json.loads(path.read_text(encoding="utf-8").strip())
    payload = dict(original)
    payload["extra"] = 1
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="keys"):
        read_rows(tmp_path / "shard")
    payload = dict(original)
    del payload["element"]
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="keys"):
        read_rows(tmp_path / "shard")


def test_metadata_and_prepared_state_round_trip(tmp_path: Path) -> None:
    """Observation metadata and the prepared mask state survive the codec."""
    from tools.ml_models.dataset.raw import ConditionTag, ObservationMetadata

    row = RowRecord(
        tile_id="t0",
        group_id="g0",
        frame_id=None,
        grid_rc=None,
        bin_id="low",
        element="rot90",
        metadata=ObservationMetadata(
            observation_id="obs",
            acquired_at_utc="2020-02-29T10:56:41Z",
            conditions=(ConditionTag(name="weather", value="recorded-clear"),),
            annotation_source="labels/obs.json",
            annotation_version="rev2",
            source_annotation_state="NONEMPTY",
        ),
        prepared_mask_state="EMPTY",
    )
    path = _write_one_row(tmp_path / "shard", row)
    payload = json.loads(path.read_text(encoding="utf-8").strip())
    assert payload["metadata"]["observation_id"] == "obs"
    assert payload["metadata"]["acquired_at_utc"] == "2020-02-29T10:56:41Z"
    assert payload["prepared_mask_state"] == "EMPTY"
    loaded = read_rows(tmp_path / "shard")
    assert loaded[0] == row


def test_schema2_rows_decode_with_unknown_metadata(tmp_path: Path) -> None:
    """Schema-2 rows without the new fields decode as fully unknown."""
    from tools.ml_models.dataset.raw import ObservationMetadata

    row = RowRecord(
        tile_id="t0",
        group_id="g0",
        frame_id="f0",
        grid_rc=(0, 0),
        bin_id="",
        element="id",
    )
    path = _write_one_row(tmp_path / "shard", row)
    payload = json.loads(path.read_text(encoding="utf-8").strip())
    del payload["metadata"]
    del payload["prepared_mask_state"]
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    loaded = read_rows(tmp_path / "shard")
    assert loaded[0].metadata == ObservationMetadata()
    assert loaded[0].metadata.observation_id is None
    assert loaded[0].metadata.source_annotation_state == "UNKNOWN"
    assert loaded[0].prepared_mask_state == "UNKNOWN"


def test_metadata_and_state_are_strictly_validated(tmp_path: Path) -> None:
    """Malformed metadata or a bad prepared state is rejected on read."""
    row = RowRecord(
        tile_id="t0",
        group_id="g0",
        frame_id=None,
        grid_rc=None,
        bin_id="",
        element="id",
    )
    path = _write_one_row(tmp_path / "shard", row)
    original = json.loads(path.read_text(encoding="utf-8").strip())
    bad_metadata: list[object] = [
        None,
        "obs",
        {"observation_id": "obs", "extra": 1},
        {"observation_id": ""},
        {"observation_id": "obs", "acquired_at_utc": "2020-01-01T00-00-00.000Z"},
        {"observation_id": "obs", "acquired_at_utc": "yesterday"},
        {"observation_id": "obs", "conditions": [{"name": "weather", "value": 1}]},
        {
            "observation_id": "obs",
            "conditions": [
                {"name": "weather", "value": "clear"},
                {"name": "weather", "value": "cloudy"},
            ],
        },
        {"observation_id": "obs", "source_annotation_state": "PARTIAL"},
    ]
    for bad in bad_metadata:
        payload = dict(original)
        payload["metadata"] = bad
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        with pytest.raises(ValueError):
            read_rows(tmp_path / "shard")
    for bad_state in (None, "PARTIAL", "nonempty", 1):
        payload = dict(original)
        payload["prepared_mask_state"] = bad_state
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        with pytest.raises(ValueError):
            read_rows(tmp_path / "shard")


def test_metadata_write_boundary_is_validated(tmp_path: Path) -> None:
    """An invalid metadata value or state fails when the rows are serialized."""
    from tools.ml_models.dataset.raw import ObservationMetadata

    image = np.zeros((3, 4, 8), dtype=np.float32)
    gsd = np.array([10.0, 20.0], dtype=np.float32)
    row = RowRecord(
        tile_id="t0",
        group_id="g0",
        frame_id=None,
        grid_rc=None,
        bin_id="",
        element="id",
    )
    bad = ObservationMetadata(source_annotation_state="MISSING")
    object.__setattr__(bad, "source_annotation_state", "BOGUS")
    writer = ShardWriter(tmp_path / "a", 1, 4, 8, channels=3, with_masks=False)
    writer.append(image, gsd, 0.0, None, replace(row, metadata=bad))
    with pytest.raises(ValueError):
        writer.close()
    writer.abort()
    writer = ShardWriter(tmp_path / "b", 1, 4, 8, channels=3, with_masks=False)
    writer.append(image, gsd, 0.0, None, replace(row, prepared_mask_state=cast(Any, "bogus")))
    with pytest.raises(ValueError):
        writer.close()
    writer.abort()


def test_metadata_change_alters_dataset_hash(tmp_path: Path) -> None:
    """Mutating only the metadata row field changes the content hash."""
    from tools.ml_models.dataset.manifest import compute_dataset_hash
    from tools.ml_models.dataset.raw import ObservationMetadata

    base = RowRecord(
        tile_id="t0",
        group_id="g0",
        frame_id=None,
        grid_rc=None,
        bin_id="",
        element="id",
    )
    _write_one_row(tmp_path / "a", base)
    _write_one_row(
        tmp_path / "b",
        replace(base, metadata=ObservationMetadata(observation_id="obs")),
    )
    assert compute_dataset_hash(tmp_path / "a") != compute_dataset_hash(tmp_path / "b")
