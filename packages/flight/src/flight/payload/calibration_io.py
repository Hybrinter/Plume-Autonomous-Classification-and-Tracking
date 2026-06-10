"""Startup-time loading of mosaic calibration artifacts (checksummed .npy files).

Lives outside preprocess/ because it performs file I/O; the preprocess package stays
pure (no I/O, no global state). manifest.json maps each artifact name (dark_frame,
flat_field, bad_pixel_mask) to {"file": <name.npy>, "sha256": <hex digest>}.

Loading procedure:
    1. Parse manifest.json in calibration_dir.
    2. For each of the three required artifacts, read the .npy file and verify its
       sha256 digest against the manifest entry.
    3. Verify that each artifact shape matches (height_px, width_px).
    4. Return a MosaicCalibration on success.

Any missing file, checksum mismatch, wrong shape, malformed manifest, or missing
directory yields Err(CALIBRATION_INVALID). The composition root treats that as an
unrecoverable startup failure (raises SystemExit).

build_identity_calibration is the SIL/dev fallback, selected by
SensorConfig.calibration_dir == "". It MUST NOT be used in flight (artifacts from
sensor characterization are required).

Satisfies: REQ-AIML-PREP-002.
"""

from __future__ import annotations

# stdlib
import hashlib
import json
from pathlib import Path

# third-party
import numpy as np

# internal
from flight.libs.types import Err, FaultCode, Ok, Result
from flight.payload.preprocess import MosaicCalibration

_ARTIFACT_NAMES: tuple[str, ...] = ("dark_frame", "flat_field", "bad_pixel_mask")


def build_identity_calibration(height_px: int, width_px: int) -> MosaicCalibration:
    """Build an identity MosaicCalibration for SIL and development use only.

    Returns a calibration with zero dark signal, unit flat field, and no bad pixels,
    which leaves the raw mosaic values completely unchanged after calibrate_mosaic.
    This is NOT suitable for flight -- sensor-characterization artifacts are required.

    Args:
        height_px: Sensor mosaic height in pixels (must match SensorConfig.height_px).
        width_px: Sensor mosaic width in pixels (must match SensorConfig.width_px).

    Returns:
        MosaicCalibration with dark_frame all-zeros, flat_field all-ones, and
        bad_pixel_mask all-False, each of shape (height_px, width_px).

    Notes:
        With identity calibration, calibrate_mosaic returns the raw DN values cast to
        float32, unchanged. Downstream normalization still clips and scales to [0, 1].
    """
    shape = (height_px, width_px)
    return MosaicCalibration(
        dark_frame=np.zeros(shape, dtype=np.float32),  # np.ndarray[float32, (H, W)]
        flat_field=np.ones(shape, dtype=np.float32),  # np.ndarray[float32, (H, W)]
        bad_pixel_mask=np.zeros(shape, dtype=bool),  # np.ndarray[bool, (H, W)]
    )


def load_calibration(
    calibration_dir: str,
    height_px: int,
    width_px: int,
) -> Result[MosaicCalibration, FaultCode]:
    """Load and verify dark/flat/bad-pixel artifacts from calibration_dir.

    Reads manifest.json to discover each artifact file and its expected sha256 digest.
    Each .npy file is read, its digest is computed and compared against the manifest,
    and its shape is verified against (height_px, width_px). All three steps must pass
    for all three artifacts before a MosaicCalibration is returned.

    Args:
        calibration_dir: Path to the directory containing manifest.json and the .npy
            artifact files. An empty string or nonexistent path yields
            Err(CALIBRATION_INVALID).
        height_px: Expected mosaic height in pixels (from SensorConfig.height_px).
        width_px: Expected mosaic width in pixels (from SensorConfig.width_px).

    Returns:
        Ok(MosaicCalibration) -- all artifacts pass integrity and shape checks.
        Err(FaultCode.CALIBRATION_INVALID) -- on any of: directory or manifest
            missing/unreadable, malformed manifest JSON, missing artifact key or
            subfield, missing .npy file, sha256 mismatch, wrong array shape.

    Notes:
        The sha256 is computed over the raw bytes of the .npy file (including the
        numpy format header), matching the digest written by _write_artifacts in the
        test helper and by any conformant artifact packaging tool.
        Arrays are cast to their canonical dtypes (float32 for dark/flat, bool for
        bad_pixel_mask) so callers are insensitive to how the artifacts were saved.
    """
    base = Path(calibration_dir)
    manifest_path = base / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        return Err(FaultCode.CALIBRATION_INVALID)

    arrays: dict[str, np.ndarray] = {}
    for name in _ARTIFACT_NAMES:
        entry = manifest.get(name)
        if not isinstance(entry, dict) or "file" not in entry or "sha256" not in entry:
            return Err(FaultCode.CALIBRATION_INVALID)
        fpath = base / str(entry["file"])
        try:
            blob = fpath.read_bytes()
        except OSError:
            return Err(FaultCode.CALIBRATION_INVALID)
        if hashlib.sha256(blob).hexdigest() != entry["sha256"]:
            return Err(FaultCode.CALIBRATION_INVALID)
        arrays[name] = np.load(fpath)

    if any(arrays[n].shape != (height_px, width_px) for n in _ARTIFACT_NAMES):
        return Err(FaultCode.CALIBRATION_INVALID)

    return Ok(
        MosaicCalibration(
            dark_frame=arrays["dark_frame"].astype(np.float32),
            flat_field=arrays["flat_field"].astype(np.float32),
            bad_pixel_mask=arrays["bad_pixel_mask"].astype(bool),
        )
    )
