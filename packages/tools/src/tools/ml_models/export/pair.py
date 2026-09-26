"""Flight-shape check and the classifier plus segmentor deploy blob.

A run is flight-promotable when its channel count, band names, and spatial
size match ``InferenceConfig``. Radiometry and ``ingest_path`` are recorded on
sidecars and do not affect the check. ``write_pair_manifest`` writes the pair
JSON the model-deploy path stores: ``version``, classifier input and output,
and segmentor input and output.

Contains:
  - flight_promotable: true when a run matches the flight input contract.
  - write_pair_manifest: write the pair JSON for two accepted sidecars.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

from flight.libs.config import InferenceConfig

from tools.ml_models.export.accept import Manifest, Shape, load_manifest

_RUN_MARKERS = ("config.toml", "summary.json")


def _as_int(value: object) -> int | None:
    """Return ``value`` when it is an int and not a bool."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _as_bands(value: object) -> tuple[str, ...] | None:
    """Return a non-empty tuple of band names, or None."""
    if not isinstance(value, list | tuple) or not value:
        return None
    names: list[str] = []
    for item in value:
        if not isinstance(item, str):
            return None
        names.append(item)
    return tuple(names)


def _pair_ints(value: object) -> tuple[int, int] | None:
    """Return a length-2 int sequence, or None."""
    if not isinstance(value, list | tuple) or len(value) != 2:
        return None
    first = _as_int(value[0])
    second = _as_int(value[1])
    if first is None or second is None:
        return None
    return first, second


def _read_json(path: Path) -> dict[str, object]:
    """Return a JSON object, or an empty mapping when the file is absent."""
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    return {str(key): value for key, value in data.items()}


def _read_toml(path: Path) -> dict[str, object]:
    """Return a TOML table, or an empty mapping when the file is absent."""
    if not path.is_file():
        return {}
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return {str(key): value for key, value in data.items()}


def _load_checkpoint(path: Path) -> dict[str, object] | None:
    """Return a checkpoint dict, or None when the file is missing or unreadable."""
    if not path.is_file():
        return None
    import torch

    try:
        try:
            raw = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            raw = torch.load(path, map_location="cpu")
    except OSError, RuntimeError, ValueError:
        return None
    if not isinstance(raw, dict):
        return None
    return {str(key): value for key, value in raw.items()}


def _checkpoint_file(run_dir: Path) -> Path | None:
    """Return ``best.pt`` or ``last.pt`` under ``run_dir``, if one exists."""
    best = run_dir / "checkpoints" / "best.pt"
    if best.is_file():
        return best
    last = run_dir / "checkpoints" / "last.pt"
    if last.is_file():
        return last
    return None


def _spatial(summary: dict[str, object], config: dict[str, object]) -> tuple[int, int] | None:
    """Return the run's flight-frame size.

    Summary ``input_height_px`` and ``input_width_px`` win when both are set.
    Otherwise a summary ``frame_hw`` or a config ``[canvas] frame_hw`` is the
    frame. The last fallback is the config crop size.
    """
    summary_h = _as_int(summary.get("input_height_px"))
    summary_w = _as_int(summary.get("input_width_px"))
    if summary_h is not None and summary_w is not None:
        return summary_h, summary_w
    frame = _pair_ints(summary.get("frame_hw"))
    if frame is not None:
        return frame
    canvas = config.get("canvas")
    if isinstance(canvas, dict):
        frame = _pair_ints(canvas.get("frame_hw"))
        if frame is not None:
            return frame
    config_h = _as_int(config.get("input_height_px"))
    config_w = _as_int(config.get("input_width_px"))
    if config_h is not None and config_w is not None:
        return config_h, config_w
    return None


def _run_dir_containing(path: Path) -> Path | None:
    """Return the nearest parent that holds ``config.toml`` or ``summary.json``."""
    for parent in path.parents:
        if any((parent / name).is_file() for name in _RUN_MARKERS):
            return parent
    return None


def flight_promotable(run_dir: str | Path, inference: InferenceConfig | None = None) -> bool:
    """Return True when ``run_dir`` matches the flight input contract.

    Args:
        run_dir: Training run directory with ``config.toml`` and/or
            ``summary.json``.
        inference: Flight input contract. None uses ``InferenceConfig()``.

    Returns:
        bool: True only when ``in_channels`` equals ``len(input_bands)``,
        ``band_names`` equal ``input_bands``, and the run spatial size equals
        ``input_height_px`` by ``input_width_px``. Radiometry and
        ``ingest_path`` do not affect the result.
    """
    root = Path(run_dir)
    summary = _read_json(root / "summary.json")
    config = _read_toml(root / "config.toml")
    channels = _as_int(summary.get("in_channels"))
    if channels is None:
        channels = _as_int(config.get("in_channels"))
    bands = _as_bands(summary.get("band_names"))
    if bands is None:
        bands = _as_bands(config.get("band_names"))
    if channels is None or bands is None:
        checkpoint_path = _checkpoint_file(root)
        payload = _load_checkpoint(checkpoint_path) if checkpoint_path is not None else None
        if payload is not None:
            if channels is None:
                channels = _as_int(payload.get("in_channels"))
            if bands is None:
                bands = _as_bands(payload.get("band_names"))
    spatial = _spatial(summary, config)
    if channels is None or bands is None or spatial is None:
        return False
    spec = inference if inference is not None else InferenceConfig()
    height, width = spatial
    return (
        channels == len(spec.input_bands)
        and bands == tuple(spec.input_bands)
        and height == spec.input_height_px
        and width == spec.input_width_px
    )


def _concrete_shape(shape: Shape, label: str) -> tuple[int, ...]:
    """Return ``shape`` as ints.

    Raises:
        ValueError: If any dimension is dynamic.
    """
    dims: list[int] = []
    for dim in shape:
        if dim is None:
            raise ValueError(f"{label} shape is not concrete: {shape}")
        dims.append(dim)
    return tuple(dims)


def write_pair_manifest(
    classifier_sidecar: str | Path,
    segmentor_sidecar: str | Path,
    dest: str | Path,
) -> dict[str, object]:
    """Write the deploy pair JSON for two accepted sidecars.

    Args:
        classifier_sidecar: Classifier manifest JSON path. A parent directory
            must be the classifier run.
        segmentor_sidecar: Segmentor manifest JSON path. A parent directory
            must be the segmentor run.
        dest: Destination JSON path.

    Returns:
        dict[str, object]: The object written to ``dest``. Keys are
        ``version``, ``classifier``, and ``segmentor``. Each network object
        has ``input_shape`` and ``output_shape``.

    Raises:
        ValueError: If either run is not flight-promotable, the sidecars
            disagree, or a shape does not match ``InferenceConfig``.
        FileNotFoundError: If a sidecar is missing.
        OSError / json.JSONDecodeError / KeyError: If a sidecar is malformed.
    """
    classifier_path = Path(classifier_sidecar)
    segmentor_path = Path(segmentor_sidecar)
    classifier = load_manifest(str(classifier_path))
    segmentor = load_manifest(str(segmentor_path))
    _require_promotable("classifier", classifier_path)
    _require_promotable("segmentor", segmentor_path)
    spec = InferenceConfig()
    expected_in = (1, len(spec.input_bands), spec.input_height_px, spec.input_width_px)
    expected_cls = (1, 1)
    expected_seg = (1, 1, spec.input_height_px, spec.input_width_px)
    _require_contract(classifier, "classifier", expected_in, expected_cls)
    _require_contract(segmentor, "segmentor", expected_in, expected_seg)
    if classifier.version != segmentor.version:
        raise ValueError(
            f"sidecar versions disagree: {classifier.version!r} != {segmentor.version!r}"
        )
    payload: dict[str, object] = {
        "version": classifier.version,
        "classifier": {
            "input_shape": list(expected_in),
            "output_shape": list(expected_cls),
        },
        "segmentor": {
            "input_shape": list(expected_in),
            "output_shape": list(expected_seg),
        },
    }
    destination = Path(dest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _require_promotable(role: str, sidecar: Path) -> None:
    """Raise ValueError when the sidecar's run is missing or not promotable."""
    run_dir = _run_dir_containing(sidecar)
    if run_dir is None:
        raise ValueError(f"no run directory for {role} sidecar: {sidecar}")
    if not flight_promotable(run_dir):
        raise ValueError(f"{role} run is not flight promotable: {run_dir}")


def _require_contract(
    manifest: Manifest,
    role: str,
    expected_input: tuple[int, ...],
    expected_output: tuple[int, ...],
) -> None:
    """Raise ValueError when one sidecar does not match the flight shapes."""
    actual_in = _concrete_shape(manifest.input_shape, role)
    actual_out = _concrete_shape(manifest.output_shape, role)
    if actual_in != expected_input or actual_out != expected_output:
        raise ValueError(
            f"{role} sidecar shape {actual_in} -> {actual_out} "
            f"does not match the flight contract {expected_input} -> {expected_output}"
        )
