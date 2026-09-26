"""Flight promotion and the classifier plus segmentor pair blob."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from flight.libs.config import InferenceConfig
from tools.cli import main as root_main
from tools.ml_models.cli import main as ml_main
from tools.ml_models.export.onnx import resolve_export_hw
from tools.ml_models.export.pair import flight_promotable, write_pair_manifest

_BANDS = ("BLUE", "GREEN", "RED")
_FLIGHT_IN = [1, 3, 1544, 2064]


def _run(
    dest: Path,
    *,
    channels: int = 3,
    bands: tuple[str, ...] = _BANDS,
    height: int = 1544,
    width: int = 2064,
    canvas_frame: tuple[int, int] | None = None,
    crop: tuple[int, int] | None = None,
    summary_hw: tuple[int, int] | None = None,
    radiometry: str = "s2_l2a_reflectance",
    ingest_path: str = "sentinel2_4250706_prism_proxy",
) -> Path:
    """Write a tiny run directory with config.toml and summary.json."""
    dest.mkdir(parents=True, exist_ok=True)
    crop_h, crop_w = crop if crop is not None else (height, width)
    lines = [
        f"in_channels = {channels}",
        f"input_height_px = {crop_h}",
        f"input_width_px = {crop_w}",
    ]
    if canvas_frame is not None:
        lines.extend(
            [
                "[canvas]",
                f"frame_hw = [{canvas_frame[0]}, {canvas_frame[1]}]",
                "window_px = 512",
            ]
        )
    (dest / "config.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary: dict[str, object] = {
        "in_channels": channels,
        "band_names": list(bands),
        "ingest_path": ingest_path,
        "radiometry": radiometry,
    }
    if summary_hw is not None:
        summary["input_height_px"] = summary_hw[0]
        summary["input_width_px"] = summary_hw[1]
    elif canvas_frame is not None:
        summary["frame_hw"] = [canvas_frame[0], canvas_frame[1]]
        summary["window_px"] = 512
    else:
        summary["input_height_px"] = height
        summary["input_width_px"] = width
    (dest / "summary.json").write_text(json.dumps(summary) + "\n", encoding="utf-8")
    return dest


def _sidecar(
    path: Path,
    *,
    input_shape: list[int],
    output_shape: list[int],
    version: str = "v1",
) -> Path:
    """Write one manifest sidecar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": version,
                "model_repo_sha": "abc",
                "dataset_hash": "ds",
                "input_shape": input_shape,
                "output_shape": output_shape,
                "sha256": "0" * 64,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_flight_frame_is_promotable(tmp_path: Path) -> None:
    """BLUE, GREEN, RED at 3 by 1544 by 2064 is promotable, including a prism proxy."""
    direct = _run(tmp_path / "direct")
    canvas = _run(
        tmp_path / "canvas",
        canvas_frame=(1544, 2064),
        crop=(256, 256),
    )
    assert flight_promotable(direct) is True
    assert flight_promotable(canvas) is True
    assert flight_promotable(canvas, InferenceConfig()) is True


def test_research_shapes_are_not_promotable(tmp_path: Path) -> None:
    """A 76 px frame, a 512 px frame, or 4 channels is not promotable."""
    small = _run(tmp_path / "small", height=76, width=76)
    window = _run(tmp_path / "window", height=512, width=512)
    wide = _run(
        tmp_path / "wide",
        channels=4,
        bands=("B1", "B2", "B3", "B4"),
    )
    assert flight_promotable(small) is False
    assert flight_promotable(window) is False
    assert flight_promotable(wide) is False
    cls = _sidecar(
        small / "classifier.json",
        input_shape=[1, 3, 76, 76],
        output_shape=[1, 1],
    )
    seg = _sidecar(
        small / "segmentor.json",
        input_shape=[1, 3, 76, 76],
        output_shape=[1, 1, 76, 76],
    )
    with pytest.raises(ValueError, match="not flight promotable"):
        write_pair_manifest(cls, seg, tmp_path / "pair.json")


def test_pair_blob_matches_flight_contract(tmp_path: Path) -> None:
    """The pair JSON records version and both network contracts."""
    run = _run(tmp_path / "run", canvas_frame=(1544, 2064), crop=(512, 512))
    cls = _sidecar(run / "classifier.json", input_shape=_FLIGHT_IN, output_shape=[1, 1])
    seg = _sidecar(
        run / "segmentor.json",
        input_shape=_FLIGHT_IN,
        output_shape=[1, 1, 1544, 2064],
    )
    payload = write_pair_manifest(cls, seg, run / "pair.json")
    written = json.loads((run / "pair.json").read_text(encoding="utf-8"))
    assert set(payload) == {"version", "classifier", "segmentor"}
    assert written["version"] == "v1"
    assert written["classifier"]["input_shape"] == _FLIGHT_IN
    assert written["classifier"]["output_shape"] == [1, 1]
    assert written["segmentor"]["input_shape"] == _FLIGHT_IN
    assert written["segmentor"]["output_shape"] == [1, 1, 1544, 2064]


def test_promotable_trace_uses_flight_frame_not_window() -> None:
    """A promotable run traces the flight frame even when the checkpoint is 512."""
    spec = InferenceConfig()
    height, width = resolve_export_hw(
        promotable=True,
        override_spatial=False,
        config_height=512,
        config_width=512,
        checkpoint_height=512,
        checkpoint_width=512,
        flight_height=spec.input_height_px,
        flight_width=spec.input_width_px,
    )
    assert (height, width) == (spec.input_height_px, spec.input_width_px)
    assert (height, width) == (1544, 2064)


def test_research_trace_keeps_checkpoint_hw() -> None:
    """A research checkpoint traces its own height and width."""
    height, width = resolve_export_hw(
        promotable=False,
        override_spatial=False,
        config_height=256,
        config_width=256,
        checkpoint_height=76,
        checkpoint_width=76,
        flight_height=1544,
        flight_width=2064,
    )
    assert (height, width) == (76, 76)
    overridden = resolve_export_hw(
        promotable=False,
        override_spatial=True,
        config_height=48,
        config_width=64,
        checkpoint_height=76,
        checkpoint_width=76,
        flight_height=1544,
        flight_width=2064,
    )
    assert overridden == (48, 64)


def test_ml_models_help_commands() -> None:
    """Module and root help for ml-models both exit 0."""
    assert ml_main(["--help"]) == 0
    assert root_main(["ml-models", "--help"]) == 0
