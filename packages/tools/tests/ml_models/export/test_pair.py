"""Flight promotion and the classifier plus segmentor pair blob."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from flight.libs.config import InferenceConfig
from tools.cli import main as root_main
from tools.ml_models.arch.registry import build
from tools.ml_models.cli import main as ml_main
from tools.ml_models.data.prism import FRAME_HW, TILE_GRID, TILE_HW
from tools.ml_models.export.onnx import ExportConfig, export, resolve_export_hw
from tools.ml_models.export.pair import flight_promotable, write_pair_manifest

_BANDS = ("BLUE", "GREEN", "RED")
_TILE_IN = [None, 3, TILE_HW[0], TILE_HW[1]]
_HAS_ONNX = importlib.util.find_spec("onnx") is not None
_skip_no_onnx = pytest.mark.skipif(not _HAS_ONNX, reason="onnx extra not installed")


def _run(
    dest: Path,
    *,
    channels: int = 3,
    bands: tuple[str, ...] = _BANDS,
    kind: str = "classifier",
    arch: str = "pactnet",
    tile_hw: tuple[int, int] | None = TILE_HW,
    height: int | None = None,
    width: int | None = None,
    frame_hw: tuple[int, int] | None = None,
    radiometry: str = "s2_l2a_reflectance",
    ingest_path: str = "sentinel2_4250706_prism_proxy",
) -> Path:
    """Write a tiny run directory with config.toml and summary.json."""
    dest.mkdir(parents=True, exist_ok=True)
    crop_h = height if height is not None else (tile_hw[0] if tile_hw is not None else 256)
    crop_w = width if width is not None else (tile_hw[1] if tile_hw is not None else 256)
    lines = [
        f'kind = "{kind}"',
        f'arch = "{arch}"',
        f"in_channels = {channels}",
        f"input_height_px = {crop_h}",
        f"input_width_px = {crop_w}",
    ]
    if frame_hw is not None:
        lines.append(f"frame_hw = [{frame_hw[0]}, {frame_hw[1]}]")
    (dest / "config.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary: dict[str, object] = {
        "kind": kind,
        "arch": arch,
        "in_channels": channels,
        "band_names": list(bands),
        "ingest_path": ingest_path,
        "radiometry": radiometry,
    }
    if tile_hw is not None:
        summary["tile_hw"] = [tile_hw[0], tile_hw[1]]
    if frame_hw is not None:
        summary["frame_hw"] = [frame_hw[0], frame_hw[1]]
    if height is not None and width is not None:
        summary["input_height_px"] = height
        summary["input_width_px"] = width
    (dest / "summary.json").write_text(json.dumps(summary) + "\n", encoding="utf-8")
    return dest


def _sidecar(
    path: Path,
    *,
    input_shape: list[int | None],
    output_shape: list[int | None],
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


def test_flight_tile_is_promotable(tmp_path: Path) -> None:
    """BLUE, GREEN, RED at the 193 by 258 tile is promotable for both flight archs."""
    classifier = _run(tmp_path / "classifier", kind="classifier", arch="pactnet")
    segmentor = _run(tmp_path / "segmentor", kind="segmentor", arch="dilatenet")
    assert TILE_HW == (193, 258)
    assert flight_promotable(classifier) is True
    assert flight_promotable(segmentor) is True
    assert flight_promotable(classifier, InferenceConfig()) is True
    assert flight_promotable(segmentor, InferenceConfig()) is True


def test_research_shapes_and_archs_are_not_promotable(tmp_path: Path) -> None:
    """A 76 px run, a 512 px run, a sensor frame, or the wrong arch is not promotable."""
    small = _run(tmp_path / "small", tile_hw=(76, 76), height=76, width=76)
    window = _run(tmp_path / "window", tile_hw=(512, 512), height=512, width=512)
    frame = _run(
        tmp_path / "frame",
        tile_hw=None,
        height=1544,
        width=2064,
        frame_hw=FRAME_HW,
        arch="pactnet",
    )
    wide = _run(tmp_path / "wide", channels=4, bands=("B1", "B2", "B3", "B4"))
    shuffle = _run(tmp_path / "shuffle", arch="shufflenetv2_x0_5")
    unet = _run(tmp_path / "unet", kind="segmentor", arch="unet")
    assert flight_promotable(small) is False
    assert flight_promotable(window) is False
    assert flight_promotable(frame) is False
    assert flight_promotable(wide) is False
    assert flight_promotable(shuffle) is False
    assert flight_promotable(unet) is False
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


def test_checkpoint_tile_hw_is_promotable(tmp_path: Path) -> None:
    """tile_hw, bands, and arch on the checkpoint promote a run when the summary omits them."""
    run = tmp_path / "ckpt"
    run.mkdir()
    (run / "config.toml").write_text(
        'kind = "classifier"\narch = ""\nin_channels = 3\n',
        encoding="utf-8",
    )
    (run / "summary.json").write_text(
        json.dumps({"kind": "classifier", "radiometry": "normalize_dn"}) + "\n",
        encoding="utf-8",
    )
    ckpt = run / "checkpoints"
    ckpt.mkdir()
    torch.save(
        {
            "kind": "classifier",
            "arch": "pactnet",
            "in_channels": 3,
            "band_names": list(_BANDS),
            "tile_hw": TILE_HW,
            "ingest_path": "flight_camera",
        },
        ckpt / "best.pt",
    )
    assert flight_promotable(run) is True


def test_pair_blob_matches_flight_tile(tmp_path: Path) -> None:
    """The pair JSON records a dynamic batch, the tile, the grid, and the camera frame."""
    classifier = _run(tmp_path / "classifier", kind="classifier", arch="pactnet")
    segmentor = _run(tmp_path / "segmentor", kind="segmentor", arch="dilatenet")
    cls = _sidecar(classifier / "classifier.json", input_shape=_TILE_IN, output_shape=[None, 1])
    seg = _sidecar(
        segmentor / "segmentor.json",
        input_shape=_TILE_IN,
        output_shape=[None, 1, TILE_HW[0], TILE_HW[1]],
    )
    payload = write_pair_manifest(cls, seg, tmp_path / "pair.json")
    written = json.loads((tmp_path / "pair.json").read_text(encoding="utf-8"))
    assert set(payload) == {"version", "grid", "frame_hw", "classifier", "segmentor"}
    assert written["version"] == "v1"
    assert written["grid"] == [TILE_GRID[0], TILE_GRID[1]]
    assert written["grid"] == [8, 8]
    assert written["frame_hw"] == [FRAME_HW[0], FRAME_HW[1]]
    assert written["frame_hw"] == [1544, 2064]
    assert written["classifier"]["input_shape"] == _TILE_IN
    assert written["classifier"]["output_shape"] == [None, 1]
    assert written["segmentor"]["input_shape"] == _TILE_IN
    assert written["segmentor"]["output_shape"] == [None, 1, 193, 258]
    assert "null" in (tmp_path / "pair.json").read_text(encoding="utf-8")


def test_pair_rejects_version_mismatch_and_concrete_batch(tmp_path: Path) -> None:
    """Sidecar versions must match, and a concrete batch is not the flight graph."""
    classifier = _run(tmp_path / "classifier", kind="classifier", arch="pactnet")
    segmentor = _run(tmp_path / "segmentor", kind="segmentor", arch="dilatenet")
    cls = _sidecar(
        classifier / "classifier.json",
        input_shape=_TILE_IN,
        output_shape=[None, 1],
        version="v1",
    )
    seg = _sidecar(
        segmentor / "segmentor.json",
        input_shape=_TILE_IN,
        output_shape=[None, 1, 193, 258],
        version="v2",
    )
    with pytest.raises(ValueError, match="versions disagree"):
        write_pair_manifest(cls, seg, tmp_path / "mismatch.json")
    concrete = _sidecar(
        segmentor / "concrete.json",
        input_shape=[1, 3, 193, 258],
        output_shape=[1, 1, 193, 258],
        version="v1",
    )
    with pytest.raises(ValueError, match="does not match"):
        write_pair_manifest(cls, concrete, tmp_path / "concrete.json")


def test_promotable_trace_uses_tile_not_sensor_frame() -> None:
    """A promotable run traces 193 by 258 and does not rewrite a checkpoint to 1544 by 2064."""
    height, width = resolve_export_hw(
        promotable=True,
        override_spatial=False,
        config_height=512,
        config_width=512,
        checkpoint_height=512,
        checkpoint_width=512,
    )
    assert (height, width) == TILE_HW
    assert (height, width) == (193, 258)
    assert (height, width) != (1544, 2064)


def test_research_trace_keeps_checkpoint_hw() -> None:
    """A 76 px run, a 512 px run, and a 1544 by 2064 run keep their own spatial size."""
    small = resolve_export_hw(
        promotable=False,
        override_spatial=False,
        config_height=256,
        config_width=256,
        checkpoint_height=76,
        checkpoint_width=76,
    )
    window = resolve_export_hw(
        promotable=False,
        override_spatial=False,
        config_height=256,
        config_width=256,
        checkpoint_height=512,
        checkpoint_width=512,
    )
    frame = resolve_export_hw(
        promotable=False,
        override_spatial=False,
        config_height=256,
        config_width=256,
        checkpoint_height=1544,
        checkpoint_width=2064,
    )
    assert small == (76, 76)
    assert window == (512, 512)
    assert frame == (1544, 2064)
    overridden = resolve_export_hw(
        promotable=False,
        override_spatial=True,
        config_height=48,
        config_width=64,
        checkpoint_height=76,
        checkpoint_width=76,
    )
    assert overridden == (48, 64)


@_skip_no_onnx
def test_promotable_export_traces_dynamic_batch_tile(tmp_path: Path) -> None:
    """A promotable pactnet checkpoint traces N by 3 by 193 by 258."""
    import onnx

    run = _run(tmp_path / "run", kind="classifier", arch="pactnet")
    model = build("classifier", "pactnet", 3)
    ckpt_dir = run / "checkpoints"
    ckpt_dir.mkdir()
    torch.save(
        {
            "kind": "classifier",
            "arch": "pactnet",
            "state_dict": model.state_dict(),
            "in_channels": 3,
            "input_height_px": 256,
            "input_width_px": 256,
            "tile_hw": TILE_HW,
            "band_names": list(_BANDS),
        },
        ckpt_dir / "best.pt",
    )
    onnx_path, _manifest_path, manifest = export(
        ExportConfig(
            kind="classifier",
            checkpoint_path=str(ckpt_dir / "best.pt"),
            output_path=str(run / "classifier.onnx"),
            in_channels=3,
            input_height_px=256,
            input_width_px=256,
        )
    )
    assert manifest.input_shape == (None, 3, 193, 258)
    assert manifest.output_shape == (None, 1)
    graph = onnx.load(str(onnx_path))
    dims = graph.graph.input[0].type.tensor_type.shape.dim
    assert dims[0].dim_param
    assert dims[1].dim_value == 3
    assert dims[2].dim_value == 193
    assert dims[3].dim_value == 258
    out_dims = graph.graph.output[0].type.tensor_type.shape.dim
    assert out_dims[0].dim_param
    assert out_dims[1].dim_value == 1


def test_ml_models_help_commands() -> None:
    """Module and root help for ml-models both exit 0."""
    assert ml_main(["--help"]) == 0
    assert root_main(["ml-models", "--help"]) == 0
