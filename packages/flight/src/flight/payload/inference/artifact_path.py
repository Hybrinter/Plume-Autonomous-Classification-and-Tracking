"""Resolve an FP32 ONNX path to its INT8 sibling when ``use_int8`` is set.

The sibling name is ``<stem>.int8.onnx`` next to the FP32 file. Callers still
pass the FP32 path in config. This helper does not check that the sibling
exists; session load fails if the file is missing.

Contains:
  - int8_sibling_path: ``*.int8.onnx`` path for an FP32 artifact.
  - resolve_quantized_path: FP32 path, or the INT8 sibling when requested.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

from pathlib import Path


def int8_sibling_path(fp32_path: str | Path) -> Path:
    """Return the sibling INT8 ONNX path for an FP32 artifact.

    Args:
        fp32_path: FP32 ``.onnx`` path from inference config.

    Returns:
        Path: ``<stem>.int8.onnx`` next to the FP32 file.
    """
    path = Path(fp32_path)
    return path.with_name(f"{path.stem}.int8{path.suffix}")


def resolve_quantized_path(fp32_path: str, use_int8: bool) -> str:
    """Return the ONNX path the compute axis should load.

    Args:
        fp32_path: Configured FP32 artifact path.
        use_int8: When true, return the INT8 sibling path.

    Returns:
        str: ``fp32_path`` when ``use_int8`` is false, else the sibling path.
    """
    if not use_int8:
        return fp32_path
    return str(int8_sibling_path(fp32_path))
