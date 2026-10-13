"""Prediction preview and gallery serialization into bundle artifacts.

The codec preserves supplied preview NPZ files byte-for-byte and emits a
canonical ``prediction-manifest.json`` recording every bound preview
(row, path, checksum, size, display mapping) and every frozen
``PredictionGallery`` (chosen rows, availability, selection method).
Preview-to-file consistency is verified - each preview path resolves to
exactly one file whose bytes match the recorded checksum and size, and
duplicate paths are refused. No row is picked, no identity is re-hashed
from sources, and no capture or model is read. Rendered figure
references are bound separately by the bundle writer.

Contains:
  - PredictionArtifacts: returned file bytes and references.
  - prediction_artifacts: the ``Result`` codec boundary.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace

from flight.libs.types import Err, Ok, Result

from tools.ml_models.analysis.artifacts import BundleFile, artifact_ref
from tools.ml_models.analysis.contracts import ArtifactRef, check_bundle_path
from tools.ml_models.analysis.visuals.predictions import PredictionPreviewCapture

_MANIFEST_PATH = "prediction-manifest.json"


@dataclass(frozen=True, slots=True)
class PredictionArtifacts:
    """Encoded prediction bundle members and their checksum references."""

    files: tuple[BundleFile, ...]
    references: tuple[ArtifactRef, ...]


def _canonical(data: object) -> bytes:
    """Serialize one frozen structure deterministically; NaN fails closed."""
    return json.dumps(data, sort_keys=True, allow_nan=False, separators=(",", ":")).encode("utf-8")


def prediction_artifacts(
    captured: PredictionPreviewCapture,
    *,
    prefix: str = "",
) -> Result[PredictionArtifacts, str]:
    """Encode frozen preview files and the canonical prediction manifest.

    ``prediction-manifest.json`` is the canonical serialization under
    ``{"schema_version": 1, "previews": [...], "galleries": [...]}``.
    Preview paths must be unique and resolve to exactly one supplied
    file whose size and checksum match the recorded identity; any
    inconsistency returns ``Err`` before caller output is touched. A
    nonempty ``prefix`` namespaces every returned file and reference
    path under ``prefix/...`` without changing bytes or checksums.
    """
    if prefix:
        try:
            check_bundle_path(prefix + "/payload.json")
        except ValueError as exc:
            return Err(f"unsafe artifact prefix: {exc}")
    try:
        if len({file.path for file in captured.files}) != len(captured.files):
            return Err("prediction files contain duplicate paths")
        if len({preview.path for preview in captured.previews}) != len(captured.previews):
            return Err("prediction previews contain duplicate paths")
        if len({preview.row.key for preview in captured.previews}) != len(captured.previews):
            return Err("prediction previews contain duplicate row keys")
        if {file.path for file in captured.files} != {
            preview.path for preview in captured.previews
        }:
            return Err("preview files must exactly match the preview descriptors")
        if _MANIFEST_PATH in {file.path for file in captured.files}:
            return Err("prediction files collide with the manifest path")
        bytes_by_path = {file.path: file.data for file in captured.files}
        for preview in captured.previews:
            data = bytes_by_path.get(preview.path)
            if data is None:
                return Err(f"preview {preview.path} has no supplied file bytes")
            if len(data) != preview.size_bytes:
                return Err(f"preview {preview.path} bytes differ from captured size")
            if hashlib.sha256(data).hexdigest() != preview.sha256:
                return Err(f"preview {preview.path} bytes differ from captured checksum")
        manifest_bytes = _canonical(
            {
                "schema_version": 1,
                "previews": [asdict(preview) for preview in captured.previews],
                "galleries": [asdict(gallery) for gallery in captured.galleries],
            }
        )
        files: list[BundleFile] = list(captured.files)
        refs: list[ArtifactRef] = []
        for file in captured.files:
            ref = artifact_ref(file.path, file.data, kind="REFERENCE", format="npz")
            if isinstance(ref, Err):
                return ref
            refs.append(ref.value)
        manifest_ref = artifact_ref(
            _MANIFEST_PATH,
            manifest_bytes,
            kind="REFERENCE",
            format="json",
            rows=len(captured.previews) + len(captured.galleries),
        )
        if isinstance(manifest_ref, Err):
            return manifest_ref
        refs.append(manifest_ref.value)
        files.append(BundleFile(path=_MANIFEST_PATH, data=manifest_bytes))
    except (OSError, TypeError, ValueError, OverflowError, RuntimeError) as exc:
        return Err(f"prediction artifact serialization failed: {exc}")
    if prefix:
        files = [replace(file, path=prefix + "/" + file.path) for file in files]
        refs = [replace(ref, path=prefix + "/" + ref.path) for ref in refs]
    return Ok(PredictionArtifacts(tuple(files), tuple(refs)))
