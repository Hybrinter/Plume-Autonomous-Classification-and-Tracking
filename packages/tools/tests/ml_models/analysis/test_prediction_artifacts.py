"""Prediction preview/manifest artifact serialization checks."""

import hashlib
import io
import json
from dataclasses import asdict, replace

import numpy as np
import pytest
from flight.libs.types import Err, Ok
from tools.ml_models.analysis.artifacts import BundleFile
from tools.ml_models.analysis.capture import CaptureRow
from tools.ml_models.analysis.contracts import (
    AvailabilityRecord,
    MetricSupport,
    MetricValue,
    SampleKey,
)
from tools.ml_models.analysis.prediction_artifacts import (
    PredictionArtifacts,
    prediction_artifacts,
)
from tools.ml_models.analysis.prediction_selections import (
    PredictionGallery,
    prediction_key,
)
from tools.ml_models.analysis.visuals.predictions import (
    PredictionPreview,
    PredictionPreviewCapture,
)


def _row(index: int = 0) -> CaptureRow:
    return CaptureRow(
        key=SampleKey(
            dataset_hash="a" * 64,
            task="classifier",
            split="val",
            spatial_shard=(4, 4),
            row_index=index,
            tile_id=f"tile-{index}",
            element="I",
        ),
        group_id="g",
        bin_id="bin",
        label=1.0,
        gsd_m=(2.0, 3.0),
        metrics=(
            MetricValue(
                name="logit",
                value=0.7,
                status="AVAILABLE",
                reason=None,
                support=MetricSupport(unit="IMAGE", n=1),
            ),
        ),
        failure_score=0.7,
        false_positive=False,
        false_negative=False,
        dataset_manifest_hash="b" * 64,
    )


def _preview_file(row: CaptureRow) -> tuple[PredictionPreview, BundleFile]:
    stream = io.BytesIO()
    np.savez(
        stream,
        image=np.zeros((3, 4, 4), dtype=np.float32),
        target=np.asarray([row.label], dtype=np.float32),
        logits=np.asarray([0.7], dtype=np.float32),
    )
    data = stream.getvalue()
    path = "previews/" + prediction_key(row) + ".npz"
    return (
        PredictionPreview(
            row=row,
            path=path,
            sha256=hashlib.sha256(data).hexdigest(),
            size_bytes=len(data),
            display_indices=(0, 1, 2),
            display_label="RGB (RED, GREEN, BLUE)",
        ),
        BundleFile(path=path, data=data),
    )


def _gallery(row: CaptureRow) -> PredictionGallery:
    return PredictionGallery(
        identifier="val_false_positive",
        family="false_positive",
        rows=(row,),
        availability=AvailabilityRecord(
            name="prediction_visual:val:false_positive",
            status="AVAILABLE",
        ),
        selection_method="method",
    )


def _capture() -> PredictionPreviewCapture:
    row = _row()
    preview, file = _preview_file(row)
    return PredictionPreviewCapture(
        previews=(preview,),
        files=(file,),
        galleries=(_gallery(row),),
    )


def _call(
    captured: PredictionPreviewCapture | None = None, prefix: str = ""
) -> PredictionArtifacts:
    result = prediction_artifacts(captured or _capture(), prefix=prefix)
    assert isinstance(result, Ok)
    return result.value


def test_manifest_is_canonical_and_files_preserved_byte_identical() -> None:
    captured = _capture()
    artifacts = _call(captured)
    manifest = next(file for file in artifacts.files if file.path == "prediction-manifest.json")
    expected = json.dumps(
        {
            "schema_version": 1,
            "previews": [asdict(preview) for preview in captured.previews],
            "galleries": [asdict(gallery) for gallery in captured.galleries],
        },
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode()
    assert manifest.data == expected
    preview_file = next(file for file in artifacts.files if file.path == captured.files[0].path)
    assert preview_file.data == captured.files[0].data


def test_reference_kinds_and_exact_checksums() -> None:
    artifacts = _call()
    for ref in artifacts.references:
        file = next(file for file in artifacts.files if file.path == ref.path)
        assert ref.sha256 == hashlib.sha256(file.data).hexdigest()
        assert ref.size_bytes == len(file.data)
        if ref.path.endswith(".npz"):
            assert ref.kind == "REFERENCE" and ref.format == "npz"
        else:
            assert ref.kind == "REFERENCE" and ref.format == "json"
    assert not any(file.path.endswith("summary.json") for file in artifacts.files)


def test_preview_without_bytes_or_hash_mismatch_fails() -> None:
    captured = _capture()
    missing = replace(captured, files=())
    assert isinstance(prediction_artifacts(missing), Err)
    preview = captured.previews[0]
    tampered = replace(
        captured,
        previews=(replace(preview, sha256="0" * 64),),
    )
    assert isinstance(prediction_artifacts(tampered), Err)
    resized = replace(captured, previews=(replace(preview, size_bytes=1),))
    assert isinstance(prediction_artifacts(resized), Err)


def test_duplicate_preview_or_file_paths_rejected() -> None:
    captured = _capture()
    assert isinstance(
        prediction_artifacts(
            replace(captured, previews=captured.previews * 2),
        ),
        Err,
    )
    assert isinstance(
        prediction_artifacts(replace(captured, files=captured.files * 2)),
        Err,
    )


def test_duplicate_preview_keys_rejected() -> None:
    captured = _capture()
    preview, file = _preview_file(_row(0))
    shadow = replace(preview, path="previews/shadow.npz")
    shadow_file = BundleFile(path="previews/shadow.npz", data=file.data)
    result = prediction_artifacts(
        replace(
            captured,
            previews=captured.previews + (shadow,),
            files=captured.files + (shadow_file,),
        )
    )
    assert isinstance(result, Err)
    assert "row keys" in result.error


def test_manifest_path_collision_and_unreferenced_files_rejected() -> None:
    captured = _capture()
    shadow_preview, shadow_file = _preview_file(_row(1))
    shadow_preview = replace(shadow_preview, path="prediction-manifest.json")
    shadow_file = replace(shadow_file, path="prediction-manifest.json")
    result = prediction_artifacts(
        replace(
            captured,
            previews=captured.previews + (shadow_preview,),
            files=captured.files + (shadow_file,),
        )
    )
    assert isinstance(result, Err)
    assert "manifest" in result.error
    unreferenced = BundleFile(path="previews/orphan.npz", data=b"npz")
    result = prediction_artifacts(replace(captured, files=captured.files + (unreferenced,)))
    assert isinstance(result, Err)
    assert "exactly match" in result.error


def test_helper_never_reselects_or_reads_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    def _bomb(*args: object, **kwargs: object) -> None:
        raise AssertionError("helper touched selection/source helpers")

    import tools.ml_models.analysis.prediction_selections as selections

    monkeypatch.setattr(selections, "prediction_gallery_data", _bomb)
    monkeypatch.setattr(selections, "prediction_key", _bomb)
    assert isinstance(prediction_artifacts(_capture()), Ok)


def test_prefix_namespaces_keep_bytes_and_checksums() -> None:
    default = _call()
    prefixed = _call(prefix="evidence/val")
    by_path = {file.path: file.data for file in default.files}
    for file in prefixed.files:
        assert file.path.startswith("evidence/val/")
        assert file.data == by_path[file.path.removeprefix("evidence/val/")]
    assert {ref.sha256 for ref in prefixed.references} == {ref.sha256 for ref in default.references}


@pytest.mark.parametrize("prefix", ("../escape", "/absolute", r"back\slash"))
def test_unsafe_prefix_rejected(prefix: str) -> None:
    assert isinstance(prediction_artifacts(_capture(), prefix=prefix), Err)
