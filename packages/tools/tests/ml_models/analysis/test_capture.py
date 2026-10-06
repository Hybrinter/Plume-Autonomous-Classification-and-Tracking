"""Tests for the capture-sink protocol and bounded disk sink."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, BinaryIO, cast

import numpy as np
import pytest
import tools.ml_models.analysis.capture as capture_module
from flight.libs.types import Err, Ok, Result
from tools.ml_models.analysis.capture import BoundedCaptureSink, CaptureRow, CaptureSink
from tools.ml_models.analysis.config import CaptureConfig
from tools.ml_models.analysis.contracts import ArtifactRef, MetricSupport, MetricValue, SampleKey


class _Sink:
    """Minimal structural CaptureSink implementation."""

    def add(
        self,
        row: CaptureRow,
        *,
        image: np.ndarray,
        target: np.ndarray,
        logits: np.ndarray,
    ) -> Result[None, str]:
        return Ok(None)

    def abort(self, reason: str) -> Result[None, str]:
        return Ok(None)

    def close(self) -> Result[None, str]:
        return Ok(None)

    def references(self) -> tuple[ArtifactRef, ...]:
        return ()


class _NoAdd:
    def abort(self, reason: str) -> Result[None, str]:
        return Ok(None)

    def close(self) -> Result[None, str]:
        return Ok(None)

    def references(self) -> tuple[ArtifactRef, ...]:
        return ()


class _NoReferences:
    def add(
        self,
        row: CaptureRow,
        *,
        image: np.ndarray,
        target: np.ndarray,
        logits: np.ndarray,
    ) -> Result[None, str]:
        return Ok(None)

    def abort(self, reason: str) -> Result[None, str]:
        return Ok(None)

    def close(self) -> Result[None, str]:
        return Ok(None)


def _key(row_index: int = 0, element: str = "a", shard: tuple[int, int] = (8, 8)) -> SampleKey:
    return SampleKey(
        dataset_hash="a" * 64,
        task="segmentor",
        split="val",
        spatial_shard=shard,
        row_index=row_index,
        tile_id="tile-0",
        element=element,
    )


def _metric() -> MetricValue:
    return MetricValue(
        name="foreground_iou_mean_positive_images",
        value=0.5,
        status="AVAILABLE",
        unit="dimensionless",
        aggregation="per_image_mean",
        support=MetricSupport(unit="IMAGE", n=1),
    )


def _row(
    row_index: int = 0,
    *,
    element: str = "a",
    failure_score: float = 0.5,
    false_positive: bool = False,
    false_negative: bool = False,
    shard: tuple[int, int] = (8, 8),
    label: float = 1.0,
    gsd_m: tuple[float, float] = (0.5, 0.5),
    metrics: tuple[MetricValue, ...] | None = None,
) -> CaptureRow:
    return CaptureRow(
        key=_key(row_index, element, shard),
        group_id="g0",
        bin_id="b0",
        label=label,
        gsd_m=gsd_m,
        metrics=(_metric(),) if metrics is None else metrics,
        failure_score=failure_score,
        false_positive=false_positive,
        false_negative=false_negative,
    )


def _arrays(h: int = 8, w: int = 8, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    image = rng.random((2, h, w)).astype(np.float32)
    target = np.ones((1, h, w), dtype=np.float32)
    logits = np.zeros((1, h, w), dtype=np.float32)
    return image, target, logits


def _dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "dataset"
    dataset.mkdir(parents=True)
    return dataset


def _make_sink(tmp_path: Path, cfg: CaptureConfig | None = None) -> BoundedCaptureSink:
    dataset = _dataset(tmp_path)
    return BoundedCaptureSink(tmp_path / "capture", cfg or CaptureConfig(), dataset=dataset)


def _metadata(out: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    return data


def _selected_keys(out: Path) -> list[str]:
    return [entry["key"] for entry in _metadata(out)["selected"]]


def test_capture_sink_is_runtime_checkable() -> None:
    """isinstance honours the structural protocol."""
    assert isinstance(_Sink(), CaptureSink)
    assert not isinstance(_NoAdd(), CaptureSink)
    assert not isinstance(_NoReferences(), CaptureSink)
    assert not isinstance(object(), CaptureSink)


def test_capture_sink_close_returns_result() -> None:
    """A conforming sink returns a Result from close."""
    sink = _Sink()
    assert isinstance(sink.close(), Ok)


def test_create_refuses_output_inside_dataset(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    result = BoundedCaptureSink.create(dataset / "capture", CaptureConfig(), dataset=dataset)
    assert isinstance(result, Err)
    result = BoundedCaptureSink.create(dataset, CaptureConfig(), dataset=dataset)
    assert isinstance(result, Err)


def test_create_refuses_existing_output(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    out = tmp_path / "capture"
    out.mkdir()
    result = BoundedCaptureSink.create(out, CaptureConfig(), dataset=dataset)
    assert isinstance(result, Err)


def test_create_reserves_marker_and_rows_file(tmp_path: Path) -> None:
    sink = _make_sink(tmp_path)
    out = tmp_path / "capture"
    assert (out / ".incomplete").is_file()
    assert (out / "rows.jsonl").is_file()
    assert sink.references() == ()


def test_add_streams_scalar_rows_and_close_releases_marker(tmp_path: Path) -> None:
    sink = _make_sink(tmp_path)
    out = tmp_path / "capture"
    image, target, logits = _arrays()
    assert isinstance(sink.add(_row(0), image=image, target=target, logits=logits), Ok)
    assert isinstance(sink.add(_row(1), image=image, target=target, logits=logits), Ok)
    lines = (out / "rows.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["key"]["row_index"] == 0
    assert first["array_view"] == "CANONICAL"
    assert first["metrics"][0]["name"] == "foreground_iou_mean_positive_images"
    assert first["metrics"][0]["value"] == 0.5
    assert first["failure_score"] == 0.5
    assert list(first) == sorted(first)
    assert isinstance(sink.close(), Ok)
    assert not (out / ".incomplete").exists()
    assert sink.references() != ()


def test_add_rejects_duplicate_key_then_fails_closed(tmp_path: Path) -> None:
    sink = _make_sink(tmp_path)
    image, target, logits = _arrays()
    assert isinstance(sink.add(_row(0), image=image, target=target, logits=logits), Ok)
    assert isinstance(sink.add(_row(0), image=image, target=target, logits=logits), Err)
    assert isinstance(sink.add(_row(1), image=image, target=target, logits=logits), Err)
    assert isinstance(sink.close(), Err)
    assert (tmp_path / "capture" / ".incomplete").exists()
    assert sink.references() == ()


def test_add_rejects_wrong_dtype_nonfinite_and_misaligned_arrays(tmp_path: Path) -> None:
    image, target, logits = _arrays()
    sink = _make_sink(tmp_path / "a")
    assert isinstance(
        sink.add(_row(0), image=image.astype(np.float64), target=target, logits=logits), Err
    )
    sink = _make_sink(tmp_path / "b")
    bad = image.copy()
    bad[0, 0, 0] = np.float32(np.nan)
    assert isinstance(sink.add(_row(0), image=bad, target=target, logits=logits), Err)
    sink = _make_sink(tmp_path / "c")
    assert isinstance(sink.add(_row(0), image=image, target=target[0], logits=logits), Err)
    sink = _make_sink(tmp_path / "d")
    assert isinstance(
        sink.add(_row(0), image=image, target=target, logits=np.zeros((1, 4, 4), np.float32)),
        Err,
    )


def test_add_rejects_value_domain_and_key_mismatch(tmp_path: Path) -> None:
    image, target, logits = _arrays()
    sink = _make_sink(tmp_path / "a")
    hot = image.copy()
    hot[0, 0, 0] = np.float32(1.5)
    assert isinstance(sink.add(_row(0), image=hot, target=target, logits=logits), Err)
    sink = _make_sink(tmp_path / "b")
    neg = image.copy()
    neg[0, 0, 0] = np.float32(-0.5)
    assert isinstance(sink.add(_row(0), image=neg, target=target, logits=logits), Err)
    sink = _make_sink(tmp_path / "c")
    fuzzy = target.copy()
    fuzzy[0, 0, 0] = np.float32(0.5)
    assert isinstance(sink.add(_row(0), image=image, target=fuzzy, logits=logits), Err)
    sink = _make_sink(tmp_path / "d")
    assert isinstance(
        sink.add(_row(0, shard=(4, 4)), image=image, target=target, logits=logits), Err
    )
    sink = _make_sink(tmp_path / "e")
    assert isinstance(
        sink.add(
            _row(0),
            image=np.zeros((2, 8, 0), np.float32),
            target=target,
            logits=logits,
        ),
        Err,
    )


def test_add_classifier_arrays(tmp_path: Path) -> None:
    sink = _make_sink(tmp_path)
    row = CaptureRow(
        key=SampleKey(
            dataset_hash="a" * 64,
            task="classifier",
            split="val",
            spatial_shard=(8, 8),
            row_index=0,
            tile_id="tile-0",
            element="a",
        ),
        group_id="g0",
        bin_id="b0",
        label=1.0,
        gsd_m=(0.5, 0.5),
        metrics=(_metric(),),
        failure_score=0.5,
    )
    image = np.ones((2, 8, 8), dtype=np.float32)
    target = np.array([1.0], dtype=np.float32)
    logits = np.array([0.25], dtype=np.float32)
    assert isinstance(sink.add(row, image=image, target=target, logits=logits), Ok)
    assert isinstance(sink.close(), Ok)


def test_capture_row_rejects_invalid_scalars_and_duplicate_metrics() -> None:
    with pytest.raises(ValueError):
        _row(0, failure_score=float("nan"))
    with pytest.raises(ValueError):
        _row(0, label=-1.0)
    with pytest.raises(ValueError):
        _row(0, label=0.5)
    with pytest.raises(ValueError):
        _row(0, label=2.0)
    with pytest.raises(ValueError):
        _row(0, gsd_m=(0.5, 0.0))
    with pytest.raises(ValueError):
        _row(0, gsd_m=(0.5, -0.5))
    with pytest.raises(ValueError):
        _row(0, metrics=(_metric(), _metric()))


def test_selection_deterministic_under_reversed_insertion(tmp_path: Path) -> None:
    cfg = CaptureConfig(max_preview_images=4, examples_per_family=2)
    dataset = _dataset(tmp_path)
    rows = [_row(i, element=f"e{i}", failure_score=i / 10.0) for i in range(6)]
    sink_a = BoundedCaptureSink(tmp_path / "a", cfg, dataset=dataset)
    sink_b = BoundedCaptureSink(tmp_path / "b", cfg, dataset=dataset)
    image, target, logits = _arrays()
    for row in rows:
        assert isinstance(sink_a.add(row, image=image, target=target, logits=logits), Ok)
    for row in reversed(rows):
        assert isinstance(sink_b.add(row, image=image, target=target, logits=logits), Ok)
    assert isinstance(sink_a.close(), Ok)
    assert isinstance(sink_b.close(), Ok)
    assert _selected_keys(tmp_path / "a") == _selected_keys(tmp_path / "b")


def test_selection_includes_worst_and_flagged_families(tmp_path: Path) -> None:
    sink = _make_sink(tmp_path, CaptureConfig(max_preview_images=4, examples_per_family=2))
    image, target, logits = _arrays()
    assert isinstance(
        sink.add(
            _row(0, element="worst", failure_score=0.99, false_positive=True),
            image=image,
            target=target,
            logits=logits,
        ),
        Ok,
    )
    for i in range(1, 5):
        assert isinstance(
            sink.add(_row(i, failure_score=0.1), image=image, target=target, logits=logits), Ok
        )
    assert isinstance(sink.close(), Ok)
    out = tmp_path / "capture"
    selected = list(_metadata(out)["selected"])
    families = [family for entry in selected for family in entry["families"]]
    assert "WORST" in families
    assert "FALSE_POSITIVE" in families
    flagged = [entry for entry in selected if "FALSE_POSITIVE" in entry["families"]]
    assert len(flagged) == 1


def test_selection_respects_max_preview_images_cap(tmp_path: Path) -> None:
    sink = _make_sink(tmp_path, CaptureConfig(max_preview_images=2, examples_per_family=2))
    image, target, logits = _arrays()
    for i in range(6):
        assert isinstance(
            sink.add(_row(i, failure_score=0.1 * i), image=image, target=target, logits=logits),
            Ok,
        )
    assert isinstance(sink.close(), Ok)
    out = tmp_path / "capture"
    assert len(_selected_keys(out)) <= 2
    previews = list((out / "previews").glob("*.npz")) if (out / "previews").is_dir() else []
    assert len(previews) <= 2


def test_zero_preview_budget_keeps_scalar_rows(tmp_path: Path) -> None:
    sink = _make_sink(tmp_path, CaptureConfig(max_preview_images=0, examples_per_family=0))
    image, target, logits = _arrays()
    assert isinstance(sink.add(_row(0), image=image, target=target, logits=logits), Ok)
    assert isinstance(sink.close(), Ok)
    out = tmp_path / "capture"
    assert len((out / "rows.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    assert not (out / "previews").exists()
    meta = _metadata(out)
    assert meta["seen_rows"] == 1
    assert meta["selected"] == []


def test_full_retention_writes_dense_predictions(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    cfg = CaptureConfig(retention="FULL", max_preview_images=1, examples_per_family=1)
    sink = BoundedCaptureSink(tmp_path / "capture", cfg, dataset=dataset)
    image, target, logits = _arrays()
    assert isinstance(sink.add(_row(0), image=image, target=target, logits=logits), Ok)
    assert isinstance(sink.add(_row(1), image=image, target=target, logits=logits), Ok)
    assert isinstance(sink.close(), Ok)
    full = sorted((tmp_path / "capture" / "full").glob("*.npz"))
    assert len(full) == 2
    compact_out = tmp_path / "compact"
    sink = BoundedCaptureSink(compact_out, CaptureConfig(), dataset=dataset)
    assert isinstance(sink.add(_row(0), image=image, target=target, logits=logits), Ok)
    assert isinstance(sink.close(), Ok)
    assert not (compact_out / "full").exists()


def test_npz_roundtrip_variable_shapes(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    cfg = CaptureConfig(retention="FULL")
    sink = BoundedCaptureSink(tmp_path / "capture", cfg, dataset=dataset)
    image_a, target_a, logits_a = _arrays(h=8, w=8)
    image_b, target_b, logits_b = _arrays(h=5, w=3, seed=1)
    assert isinstance(sink.add(_row(0), image=image_a, target=target_a, logits=logits_a), Ok)
    assert isinstance(
        sink.add(_row(1, shard=(5, 3)), image=image_b, target=target_b, logits=logits_b), Ok
    )
    assert isinstance(sink.close(), Ok)
    files = sorted((tmp_path / "capture" / "full").glob("*.npz"))
    loaded = [np.load(path, allow_pickle=False) for path in files]
    assert {tuple(arr["image"].shape) for arr in loaded} == {(2, 8, 8), (2, 5, 3)}
    for arr in loaded:
        assert arr["image"].dtype == np.float32
        assert arr["target"].shape[0] == 1
        assert arr["logits"].shape == arr["target"].shape


def test_scalar_budget_exhaustion_fails_with_marker(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    cfg = CaptureConfig(max_capture_bytes=256)
    sink = BoundedCaptureSink(tmp_path / "capture", cfg, dataset=dataset)
    image, target, logits = _arrays()
    result = sink.add(_row(0), image=image, target=target, logits=logits)
    close = sink.close()
    assert isinstance(result, Err) or isinstance(close, Err)
    assert (tmp_path / "capture" / ".incomplete").exists()
    assert sink.references() == ()


def test_preview_omission_under_tight_budget(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    cfg = CaptureConfig(max_preview_images=4, examples_per_family=2, max_capture_bytes=1 << 30)
    sink = BoundedCaptureSink(tmp_path / "capture", cfg, dataset=dataset)
    image, target, logits = _arrays(h=16, w=16)
    for i in range(4):
        assert isinstance(
            sink.add(
                _row(i, failure_score=0.1 * i, shard=(16, 16)),
                image=image,
                target=target,
                logits=logits,
            ),
            Ok,
        )
    sink._cfg = CaptureConfig(
        max_preview_images=4,
        examples_per_family=2,
        max_capture_bytes=sink._persisted + sink._cache_bytes + sink._reserve,
    )
    assert isinstance(sink.close(), Ok)
    out = tmp_path / "capture"
    meta = _metadata(out)
    assert len(meta["selected"]) >= 1
    assert meta["omitted_previews"] >= 1
    kept = [e for e in meta["selected"] if e["status"] == "KEPT"]
    omitted = [e for e in meta["selected"] if e["status"] == "OMITTED"]
    assert len(kept) + len(omitted) == len(meta["selected"])
    assert all(e["file"] is not None for e in kept)
    assert all(e["file"] is None and e["reason"] == "byte_budget" for e in omitted)
    assert not (out / ".incomplete").exists()
    assert sink.references() != ()


def test_close_is_idempotent_and_add_after_close_fails(tmp_path: Path) -> None:
    sink = _make_sink(tmp_path)
    image, target, logits = _arrays()
    assert isinstance(sink.add(_row(0), image=image, target=target, logits=logits), Ok)
    assert isinstance(sink.close(), Ok)
    assert isinstance(sink.close(), Ok)
    assert isinstance(sink.add(_row(1), image=image, target=target, logits=logits), Err)


def test_abort_marks_failure_and_close_reports_reason(tmp_path: Path) -> None:
    sink = _make_sink(tmp_path)
    image, target, logits = _arrays()
    assert isinstance(sink.add(_row(0), image=image, target=target, logits=logits), Ok)
    assert isinstance(sink.abort("evaluation failed upstream"), Ok)
    close = sink.close()
    assert isinstance(close, Err)
    assert "evaluation failed upstream" in close.error
    assert (tmp_path / "capture" / ".incomplete").exists()
    assert sink.references() == ()
    assert isinstance(sink.add(_row(1), image=image, target=target, logits=logits), Err)


def test_references_cover_all_persisted_files(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    cfg = CaptureConfig(retention="FULL", max_preview_images=2, examples_per_family=1)
    sink = BoundedCaptureSink(tmp_path / "capture", cfg, dataset=dataset)
    image, target, logits = _arrays()
    assert isinstance(sink.add(_row(0), image=image, target=target, logits=logits), Ok)
    assert isinstance(sink.add(_row(1), image=image, target=target, logits=logits), Ok)
    assert isinstance(sink.close(), Ok)
    refs = {ref.path: ref for ref in sink.references()}
    assert refs["rows.jsonl"].population == "evaluated_rows"
    assert refs["rows.jsonl"].rows == 2
    previews = [path for path in refs if path.startswith("previews/")]
    fulls = [path for path in refs if path.startswith("full/")]
    assert previews and all(refs[p].population == "selected_previews" for p in previews)
    assert len(fulls) == 2
    assert all(refs[f].population == "full_predictions" for f in fulls)
    out = tmp_path / "capture"
    for path, ref in refs.items():
        file = out.joinpath(*path.split("/"))
        assert file.stat().st_size == ref.size_bytes


def test_add_owns_copies_of_supplied_arrays(tmp_path: Path) -> None:
    sink = _make_sink(tmp_path)
    image, target, logits = _arrays()
    image_copy, target_copy, logits_copy = image.copy(), target.copy(), logits.copy()
    assert isinstance(sink.add(_row(0), image=image, target=target, logits=logits), Ok)
    image.fill(9.0)
    target.fill(0.0)
    logits.fill(-9.0)
    assert isinstance(sink.close(), Ok)
    previews = list((tmp_path / "capture" / "previews").glob("*.npz"))
    assert len(previews) == 1
    loaded = np.load(previews[0], allow_pickle=False)
    np.testing.assert_array_equal(loaded["image"], image_copy)
    np.testing.assert_array_equal(loaded["target"], target_copy)
    np.testing.assert_array_equal(loaded["logits"], logits_copy)


def test_mandatory_scalar_row_evicts_preview_cache(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    cfg = CaptureConfig(max_preview_images=4, examples_per_family=2, max_capture_bytes=1 << 30)
    sink = BoundedCaptureSink(tmp_path / "capture", cfg, dataset=dataset)
    image, target, logits = _arrays(h=16, w=16)
    for i in range(2):
        assert isinstance(
            sink.add(_row(i, shard=(16, 16)), image=image, target=target, logits=logits),
            Ok,
        )
    assert sink._cache_bytes > 0
    sink._cfg = CaptureConfig(
        max_preview_images=4,
        examples_per_family=2,
        max_capture_bytes=sink._persisted + sink._reserve + 1024,
    )
    assert isinstance(
        sink.add(_row(2, shard=(16, 16)), image=image, target=target, logits=logits),
        Ok,
    )
    assert isinstance(sink.close(), Ok)
    out = tmp_path / "capture"
    meta = _metadata(out)
    assert meta["omitted_previews"] >= 1
    rows = (out / "rows.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 3


def test_full_write_evicts_preview_cache(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    cfg = CaptureConfig(
        retention="FULL",
        max_preview_images=4,
        examples_per_family=2,
        max_capture_bytes=1 << 30,
    )
    sink = BoundedCaptureSink(tmp_path / "capture", cfg, dataset=dataset)
    image, target, logits = _arrays(h=16, w=16)
    for i in range(2):
        assert isinstance(
            sink.add(_row(i, shard=(16, 16)), image=image, target=target, logits=logits),
            Ok,
        )
    assert sink._cache_bytes > 0
    sink._cfg = CaptureConfig(
        retention="FULL",
        max_preview_images=4,
        examples_per_family=2,
        max_capture_bytes=sink._persisted + sink._reserve + 8000,
    )
    assert isinstance(
        sink.add(_row(2, shard=(16, 16)), image=image, target=target, logits=logits),
        Ok,
    )
    assert isinstance(sink.close(), Ok)
    out = tmp_path / "capture"
    assert len(list((out / "full").glob("*.npz"))) == 3
    meta = _metadata(out)
    assert meta["omitted_previews"] >= 1


def test_close_write_failure_keeps_marker(tmp_path: Path) -> None:
    sink = _make_sink(tmp_path)
    image, target, logits = _arrays()
    assert isinstance(sink.add(_row(0), image=image, target=target, logits=logits), Ok)
    out = tmp_path / "capture"
    (out / "previews").write_bytes(b"occupied")
    assert isinstance(sink.close(), Err)
    assert (out / ".incomplete").exists()
    assert sink.references() == ()


def test_reference_failure_keeps_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sink = _make_sink(tmp_path)
    image, target, logits = _arrays()
    assert isinstance(sink.add(_row(0), image=image, target=target, logits=logits), Ok)

    def _bad_ref(*args: object, **kwargs: object) -> Result[ArtifactRef, str]:
        return Err("checksum failed")

    monkeypatch.setattr(capture_module, "artifact_file_ref", _bad_ref)
    assert isinstance(sink.close(), Err)
    assert (tmp_path / "capture" / ".incomplete").exists()
    assert sink.references() == ()


def test_rows_handle_close_failure_is_sticky(tmp_path: Path) -> None:
    sink = _make_sink(tmp_path)
    image, target, logits = _arrays()
    assert isinstance(sink.add(_row(0), image=image, target=target, logits=logits), Ok)

    class _BadHandle:
        def close(self) -> None:
            raise OSError("stream close failed")

    sink._rows_handle = cast(BinaryIO, _BadHandle())
    assert isinstance(sink.close(), Err)
    assert isinstance(sink.close(), Err)
    assert (tmp_path / "capture" / ".incomplete").exists()
    assert sink.references() == ()


def test_close_releases_candidate_cache(tmp_path: Path) -> None:
    sink = _make_sink(tmp_path)
    image, target, logits = _arrays()
    assert isinstance(sink.add(_row(0), image=image, target=target, logits=logits), Ok)
    assert sink._cache_bytes > 0
    assert isinstance(sink.close(), Ok)
    assert sink._cache == {}
    assert sink._cache_bytes == 0


def test_unselected_candidates_do_not_consume_preview_budget(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    cfg = CaptureConfig(max_preview_images=2, examples_per_family=2, max_capture_bytes=1 << 30)
    sink = BoundedCaptureSink(tmp_path / "capture", cfg, dataset=dataset)
    image, target, logits = _arrays()
    for i in range(8):
        assert isinstance(
            sink.add(
                _row(i, failure_score=0.1 * i),
                image=image,
                target=target,
                logits=logits,
            ),
            Ok,
        )
    prospective = sink._select()
    assert len(sink._cache) > len(prospective)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, image=image, target=target, logits=logits)
    npz_size = len(buffer.getvalue())
    arrays_nbytes = int(image.nbytes + target.nbytes + logits.nbytes)
    selected_bytes = len(prospective) * (arrays_nbytes + npz_size)
    sink._cfg = CaptureConfig(
        max_preview_images=2,
        examples_per_family=2,
        max_capture_bytes=sink._persisted + sink._reserve + selected_bytes + 512,
    )
    assert isinstance(sink.close(), Ok)
    meta = _metadata(tmp_path / "capture")
    assert meta["selected"] == [entry for entry in meta["selected"] if entry["status"] == "KEPT"]
    assert meta["omitted_previews"] == 0
    assert len(meta["selected"]) == len(prospective)
