"""Classifier/segmentor pairing and flight-promotion gates.

``training_promotable`` is a metadata-only check on a finished run's
summary. ``flight_promotable`` is the strict public gate: the metadata must
pass AND a real exported artifact must validate through ``open_session``.
``write_pair_manifest`` validates both artifacts against their sidecars and
acceptance evidence, then writes a combined pair manifest. Nothing here
copies to active deployment locations.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import cast

from flight.libs.config import InferenceConfig
from flight.libs.types import Err, Ok, Result

from tools.ml_models.export.contract import (
    CONDITIONING_ID,
    coverage_ok,
    required_gsd_coverage,
)
from tools.ml_models.export.manifest import (
    ModelManifest,
    acceptance_path,
    load_manifest,
    sidecar_path,
)
from tools.ml_models.export.session import open_session

_FAMILY_PREFIX = {"classifier": "pactnet", "segmentor": "dilatenet"}


def _pairs(value: object) -> tuple[float, float] | None:
    """Coerce a two-number sequence to a finite, ordered float pair."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        low, high = float(value[0]), float(value[1])
    except TypeError, ValueError:
        return None
    if not math.isfinite(low) or not math.isfinite(high):
        return None
    return low, high


def _summary_provenance(run: Path) -> dict[str, object] | None:
    """Load ``run/summary.json`` and return its provenance plus header fields."""
    try:
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    except OSError, ValueError:
        return None
    if not isinstance(summary, dict) or not isinstance(summary.get("provenance"), dict):
        return None
    return summary


def _valid_bounds(
    provenance: dict[str, object],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Return the (min, max) GSD pairs, or None when absent or malformed."""
    gsd_min = _pairs(provenance.get("gsd_min_m"))
    gsd_max = _pairs(provenance.get("gsd_max_m"))
    if gsd_min is None or gsd_max is None:
        return None
    if any(low <= 0 or low > high for low, high in zip(gsd_min, gsd_max, strict=True)):
        return None
    return gsd_min, gsd_max


def training_promotable(
    run_dir: str | Path,
    inference: InferenceConfig | None = None,
    allow_partial_gsd: bool = False,
) -> bool:
    """Return True when run metadata satisfies the flight contract fields.

    Metadata-only: reads ``run_dir/summary.json`` and requires
    ``checkpoints/last.pt`` to exist. Verifies model family, bands, unit
    normalization, GSD reference, conditioning marker, and that actual
    training GSD coverage spans the required flight coverage (unless
    ``allow_partial_gsd``). Training tile sizes are intentionally
    unconstrained: native-only and mixed-extent runs are promotable.
    Malformed metadata returns False; this function never raises. It makes no
    claim about an exported graph — see ``flight_promotable``.
    """
    run = Path(run_dir)
    if not (run / "checkpoints" / "last.pt").is_file():
        return False
    summary = _summary_provenance(run)
    if summary is None:
        return False
    inference = inference or InferenceConfig()
    provenance = cast(dict[str, object], summary["provenance"])
    kind = summary.get("kind")
    arch = summary.get("arch")
    if (
        not isinstance(kind, str)
        or not isinstance(arch, str)
        or kind not in _FAMILY_PREFIX
        or arch.split("_")[0] != _FAMILY_PREFIX[kind]
    ):
        return False
    if summary.get("conditioning") != CONDITIONING_ID:
        return False
    bands = provenance.get("band_names")
    if (
        not isinstance(bands, (list, tuple))
        or not all(isinstance(band, str) for band in bands)
        or tuple(bands) != tuple(inference.input_bands)
    ):
        return False
    if provenance.get("norm") != "unit":
        return False
    reference = provenance.get("gsd_reference_m")
    if (
        not isinstance(reference, (int, float))
        or not math.isfinite(reference)
        or reference != inference.gsd_reference_m
    ):
        return False
    bounds = _valid_bounds(provenance)
    if bounds is None:
        return False
    gsd_min, gsd_max = bounds
    required = required_gsd_coverage()
    if required is None:
        return False
    if allow_partial_gsd:
        return True
    return coverage_ok(gsd_min, gsd_max, required)


def _manifest_matches_run(
    manifest: ModelManifest,
    summary: dict[str, object],
    inference: InferenceConfig,
) -> bool:
    """Check a sidecar against run metadata and the flight preprocessing."""
    provenance = cast(dict[str, object], summary["provenance"])
    bounds = _valid_bounds(provenance)
    if bounds is None:
        return False
    gsd_min, gsd_max = bounds
    return (
        manifest.kind == summary.get("kind")
        and manifest.arch == summary.get("arch")
        and tuple(manifest.band_names) == tuple(inference.input_bands)
        and manifest.norm == "unit"
        and manifest.gsd_reference_m == inference.gsd_reference_m
        and manifest.conditioning == CONDITIONING_ID
        and manifest.dataset_hash == summary.get("dataset_hash")
        and manifest.gsd_min_m == gsd_min
        and manifest.gsd_max_m == gsd_max
    )


def flight_promotable(
    run_dir: str | Path,
    inference: InferenceConfig | None = None,
    allow_partial_gsd: bool = False,
    *,
    artifact_path: str | Path | None = None,
) -> bool:
    """Strict flight gate: run metadata plus a validated exported graph.

    Returns True only when ``training_promotable`` passes and a real ONNX
    artifact — ``artifact_path`` when given, else every ``*.onnx`` under
    ``run_dir`` — pairs with a valid ``ModelManifest`` matching the run's
    kind, arch, bands, reference, normalization, conditioning, dataset hash,
    and provenance GSD bounds, and ``open_session`` validates the artifact.
    Without an exported graph or the onnxruntime SDK this returns False; it
    never claims promotability from checkpoint bytes alone. Never raises.
    """
    run = Path(run_dir)
    inference = inference or InferenceConfig()
    if not training_promotable(run, inference, allow_partial_gsd):
        return False
    summary = _summary_provenance(run)
    if summary is None:
        return False
    if artifact_path is not None:
        candidates = [Path(artifact_path)]
    else:
        try:
            candidates = sorted(run.rglob("*.onnx"))
        except OSError:
            return False
    for artifact in candidates:
        try:
            manifest = load_manifest(sidecar_path(artifact))
        except OSError, ValueError:
            continue
        if not _manifest_matches_run(manifest, summary, inference):
            continue
        opened = open_session(artifact, manifest)
        if isinstance(opened, Ok):
            return True
    return False


def _acceptance_ok(artifact: Path, manifest: ModelManifest) -> bool:
    """Return True when a sibling acceptance report covers this exact artifact."""
    try:
        report = json.loads(acceptance_path(artifact).read_text(encoding="utf-8"))
    except OSError, ValueError:
        return False
    return (
        isinstance(report, dict)
        and report.get("accepted") is True
        and report.get("sha256") == manifest.sha256
    )


def _artifact_entry(manifest: ModelManifest) -> dict[str, object]:
    """Build one pair-manifest artifact entry."""
    return {
        "arch": manifest.arch,
        "sha256": manifest.sha256,
        "input_names": ["image", "gsd"],
        "input_types": dict(manifest.input_types),
        "output_type": manifest.output_type,
        "input_shape": list(manifest.input_shape),
        "gsd_input_shape": list(manifest.gsd_input_shape),
        "output_shape": list(manifest.output_shape),
        "gsd_reference_m": manifest.gsd_reference_m,
        "norm": manifest.norm,
        "conditioning": manifest.conditioning,
        "gsd_encoding": manifest.gsd_encoding,
        "band_names": list(manifest.band_names),
        "gsd_min_m": list(manifest.gsd_min_m),
        "gsd_max_m": list(manifest.gsd_max_m),
    }


def _write_pair_manifest(
    classifier_sidecar: str | Path,
    segmentor_sidecar: str | Path,
    dest: str | Path,
    allow_partial_gsd: bool,
) -> dict[str, object]:
    classifier = load_manifest(classifier_sidecar)
    segmentor = load_manifest(segmentor_sidecar)
    if classifier.kind != "classifier":
        raise ValueError("classifier sidecar does not describe a classifier")
    if segmentor.kind != "segmentor":
        raise ValueError("segmentor sidecar does not describe a segmentor")
    cls_artifact = Path(classifier_sidecar).with_suffix(".onnx")
    seg_artifact = Path(segmentor_sidecar).with_suffix(".onnx")
    for artifact, manifest in ((cls_artifact, classifier), (seg_artifact, segmentor)):
        if not _acceptance_ok(artifact, manifest):
            raise ValueError(f"{artifact.name} lacks a passing acceptance report")
        opened = open_session(artifact, manifest)
        if isinstance(opened, Err):
            raise ValueError(f"{artifact.name} failed session validation: {opened.error}")
    shared = (
        "gsd_reference_m",
        "conditioning",
        "gsd_encoding",
        "norm",
        "band_names",
        "tile_hw",
        "grid",
        "frame_hw",
        "coverage_altitude_m",
    )
    for field_name in shared:
        if getattr(classifier, field_name) != getattr(segmentor, field_name):
            raise ValueError(f"paired manifests disagree on {field_name}")
    flight_reference = InferenceConfig().gsd_reference_m
    if classifier.gsd_reference_m != flight_reference:
        raise ValueError(f"pair gsd_reference_m must equal the flight reference {flight_reference}")
    required = required_gsd_coverage()
    if required is None:
        raise ValueError("required flight GSD coverage could not be computed")
    cls_covered = coverage_ok(classifier.gsd_min_m, classifier.gsd_max_m, required)
    seg_covered = coverage_ok(segmentor.gsd_min_m, segmentor.gsd_max_m, required)
    if not (cls_covered and seg_covered) and not allow_partial_gsd:
        raise ValueError(
            "paired coverage does not span required flight coverage; "
            "pass allow_partial_gsd to record the gap"
        )
    version = hashlib.sha256((classifier.sha256 + segmentor.sha256).encode()).hexdigest()[:16]
    payload: dict[str, object] = {
        "version": version,
        "grid": list(classifier.grid),
        "frame_hw": list(classifier.frame_hw),
        "tile_hw": list(classifier.tile_hw),
        "gsd_reference_m": classifier.gsd_reference_m,
        "norm": classifier.norm,
        "conditioning": classifier.conditioning,
        "gsd_encoding": classifier.gsd_encoding,
        "coverage_altitude_m": classifier.coverage_altitude_m,
        "partial_gsd": not (cls_covered and seg_covered),
        "classifier": _artifact_entry(classifier),
        "segmentor": _artifact_entry(segmentor),
    }
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with dest_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2) + "\n")
    return payload


def write_pair_manifest(
    classifier_sidecar: str | Path,
    segmentor_sidecar: str | Path,
    dest: str | Path,
    *,
    allow_partial_gsd: bool = False,
) -> Result[dict[str, object], str]:
    """Validate a classifier/segmentor pair and write the pair manifest.

    Both artifacts are hash-checked and opened through ``open_session``; both
    must carry passing acceptance reports tied to their current SHA-256, share
    the conditioning contract and preprocessing fields, carry the default
    flight GSD reference, and cover the required flight GSD range unless
    ``allow_partial_gsd`` records the gap explicitly. The destination is
    created exclusively; existing files are never overwritten.
    """
    try:
        return Ok(
            _write_pair_manifest(classifier_sidecar, segmentor_sidecar, dest, allow_partial_gsd)
        )
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        return Err(str(exc))
