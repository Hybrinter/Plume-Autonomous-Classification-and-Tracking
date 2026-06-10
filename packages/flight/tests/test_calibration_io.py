"""Tests for calibration artifact loading and the identity builder."""

import hashlib
import json
from pathlib import Path

import numpy as np
from flight.libs.types import Err, FaultCode, Ok
from flight.payload.calibration_io import build_identity_calibration, load_calibration


def _write_artifacts(tmp_path: Path, h: int, w: int, corrupt: bool = False) -> None:
    arrays = {
        "dark_frame": np.zeros((h, w), dtype=np.float32),
        "flat_field": np.ones((h, w), dtype=np.float32),
        "bad_pixel_mask": np.zeros((h, w), dtype=bool),
    }
    manifest: dict[str, dict[str, str]] = {}
    for name, arr in arrays.items():
        fpath = tmp_path / f"{name}.npy"
        np.save(fpath, arr)
        digest = hashlib.sha256(fpath.read_bytes()).hexdigest()
        if corrupt and name == "flat_field":
            digest = "0" * 64
        manifest[name] = {"file": f"{name}.npy", "sha256": digest}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_load_calibration_happy_path(tmp_path: Path) -> None:
    """Valid artifacts + matching checksums load into a MosaicCalibration."""
    _write_artifacts(tmp_path, 8, 8)
    result = load_calibration(str(tmp_path), height_px=8, width_px=8)
    assert isinstance(result, Ok)
    assert result.value.dark_frame.shape == (8, 8)
    assert result.value.bad_pixel_mask.dtype == np.bool_


def test_load_calibration_checksum_mismatch(tmp_path: Path) -> None:
    """A checksum mismatch returns Err(CALIBRATION_INVALID)."""
    _write_artifacts(tmp_path, 8, 8, corrupt=True)
    result = load_calibration(str(tmp_path), height_px=8, width_px=8)
    assert isinstance(result, Err)
    assert result.error == FaultCode.CALIBRATION_INVALID


def test_load_calibration_shape_mismatch(tmp_path: Path) -> None:
    """Artifacts whose shape disagrees with the sensor config are rejected."""
    _write_artifacts(tmp_path, 8, 8)
    result = load_calibration(str(tmp_path), height_px=16, width_px=16)
    assert isinstance(result, Err)
    assert result.error == FaultCode.CALIBRATION_INVALID


def test_load_calibration_missing_dir() -> None:
    """A nonexistent directory returns Err(CALIBRATION_INVALID)."""
    result = load_calibration("does/not/exist", height_px=8, width_px=8)
    assert isinstance(result, Err)
    assert result.error == FaultCode.CALIBRATION_INVALID


def test_identity_calibration_shape() -> None:
    """Identity calibration: zero dark, unit flat, no bad pixels."""
    cal = build_identity_calibration(height_px=8, width_px=8)
    assert float(cal.dark_frame.sum()) == 0.0
    assert float(cal.flat_field.mean()) == 1.0
    assert not cal.bad_pixel_mask.any()
