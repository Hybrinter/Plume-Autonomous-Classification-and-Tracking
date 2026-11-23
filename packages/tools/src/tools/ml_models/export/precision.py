"""FP16 and INT8 conversions of validated two-input FP32 artifacts.

``convert_fp16`` rewrites graph weights to float16 with ``keep_io_types=True``
and ``quantize_int8`` performs static QDQ quantization over real train-shard
calibration pairs. Both validate the source hash and graph through
``open_session``, derive the sidecar from the base manifest (only ``sha256``,
``version``, and ``quantization`` change), and publish the artifact plus
sidecar atomically without overwriting existing outputs. Converted artifacts
carry a new SHA-256, so acceptance evidence must be collected again before
pairing.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
from flight.libs.types import Err, Ok, Result

from tools.ml_models.export.calibration import calibration_batches
from tools.ml_models.export.export import _private_sibling, _publish
from tools.ml_models.export.manifest import (
    load_manifest,
    sidecar_path,
    write_manifest,
)
from tools.ml_models.export.session import open_session


def _write_fp16(source: Path, tmp: Path) -> None:
    """Rewrite the FP32 graph at ``source`` to FP16 with float32 I/O."""
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16

    model = onnx.load(str(source))
    onnx.save(convert_float_to_float16(model, keep_io_types=True), str(tmp))


def _write_int8(source: Path, tmp: Path, batches: list[dict[str, np.ndarray]]) -> None:
    """Write a static QDQ INT8 graph with float32 I/O at ``tmp``."""
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.calibrate import CalibrationDataReader

    class _Reader(CalibrationDataReader):  # type: ignore[misc]
        """Yield ``{"image": ..., "gsd": ...}`` calibration dicts, then None."""

        def __init__(self) -> None:
            self._index = 0

        def get_next(self) -> dict[str, np.ndarray] | None:
            if self._index >= len(batches):
                return None
            item = batches[self._index]
            self._index += 1
            return item

        def rewind(self) -> None:
            self._index = 0

    quantize_static(
        model_input=str(source),
        model_output=str(tmp),
        calibration_data_reader=_Reader(),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
    )


def _convert(
    source: str | Path,
    dest: str | Path,
    quantization: str,
    write: Callable[[Path, Path], None],
) -> Path:
    """Convert ``source`` into a new ``dest`` artifact and sidecar."""
    src = Path(source)
    dst = Path(dest)
    if src.resolve() == dst.resolve():
        raise ValueError("conversion destination must differ from the source")
    src_sidecar = sidecar_path(src)
    dst_sidecar = sidecar_path(dst)
    if not src.is_file() or not src_sidecar.is_file():
        raise FileNotFoundError(f"missing {src} or {src_sidecar}")
    if dst.exists() or dst_sidecar.exists():
        raise FileExistsError(f"refusing to overwrite {dst} or {dst_sidecar}")
    base = load_manifest(src_sidecar)
    checked = open_session(src, base)
    if isinstance(checked, Err):
        raise ValueError(f"source artifact failed validation: {checked.error}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_artifact = _private_sibling(dst)
    tmp_sidecar = _private_sibling(dst_sidecar)
    try:
        write(src, tmp_artifact)
        sha256 = hashlib.sha256(tmp_artifact.read_bytes()).hexdigest()
        derived = replace(base, sha256=sha256, version=sha256[:16], quantization=quantization)
        write_manifest(tmp_sidecar, derived)
        converted = open_session(tmp_artifact, derived)
        if isinstance(converted, Err):
            raise ValueError(f"converted artifact failed validation: {converted.error}")
        _publish(tmp_artifact, dst)
        try:
            _publish(tmp_sidecar, dst_sidecar)
        except OSError:
            # Roll back only the artifact this conversion just linked.
            if dst.exists() and os.path.samefile(dst, tmp_artifact):
                dst.unlink()
            raise
    finally:
        tmp_artifact.unlink(missing_ok=True)
        tmp_sidecar.unlink(missing_ok=True)
    return dst


def convert_fp16(source: str | Path, dest: str | Path) -> Result[Path, str]:
    """Convert a validated FP32 artifact to FP16 with float32 I/O.

    Args:
        source: FP32 ``.onnx`` artifact with a valid sidecar.
        dest: New destination artifact path. Must differ from ``source``.

    Returns:
        Result[Path, str]: Ok(destination) on success, else Err.
    """
    try:
        return Ok(_convert(source, dest, "fp16", _write_fp16))
    except (OSError, ValueError, RuntimeError, KeyError, ImportError) as exc:
        return Err(str(exc))


def quantize_int8(
    source: str | Path,
    dest: str | Path,
    *,
    dataset: str | Path,
    calib_samples: int = 32,
) -> Result[Path, str]:
    """Quantize a validated FP32 artifact to static QDQ INT8.

    Calibration consumes train-shard ``{"image", "gsd"}`` pairs collected by
    ``calibration_batches``; synthetic or held-out rows are never used.

    Args:
        source: FP32 ``.onnx`` artifact with a valid sidecar.
        dest: New destination artifact path. Must differ from ``source``.
        dataset: The finished dataset supplying calibration rows.
        calib_samples: Maximum calibration batches.

    Returns:
        Result[Path, str]: Ok(destination) on success, else Err.
    """
    src = Path(source)
    try:
        if not src.is_file() or not sidecar_path(src).is_file():
            raise FileNotFoundError(f"missing {src} or {sidecar_path(src)}")
        base = load_manifest(sidecar_path(src))
        batches = calibration_batches(dataset, base, calib_samples)
        if not batches:
            raise ValueError("INT8 calibration produced no batches")

        def write(src_path: Path, tmp: Path) -> None:
            _write_int8(src_path, tmp, batches)

        return Ok(_convert(src, dest, "int8", write))
    except (OSError, ValueError, RuntimeError, KeyError, ImportError) as exc:
        return Err(str(exc))
