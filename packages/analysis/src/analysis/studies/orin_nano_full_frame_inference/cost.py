"""FLOP and byte-traffic scaling for the Orin Nano Super study.

pact-analysis stays torch-free. Full-frame FLOPs are the locked 256-tile
G counts times ``area_scale()``. Memory traffic is input (always float32
I/O) plus weights at the named precision plus live activation maps.

Contains:
  - flops_full_frame_g: scale a 256-tile G count.
  - weight_bytes / input_bytes / activation_bytes / bytes_moved.
"""

from __future__ import annotations

from analysis.studies.orin_nano_full_frame_inference.assumptions import (
    CLS_FLOPS_TILE_G,
    CLS_PARAMS,
    FRAME_H_PX,
    FRAME_W_PX,
    IN_CHANNELS,
    LIVE_CHANNELS,
    LIVE_MAPS,
    OUTPUT_STRIDE,
    SEG_FLOPS_TILE_G,
    SEG_PARAMS,
    area_scale,
)

_ELEM_BYTES: dict[str, int] = {"fp32": 4, "fp16": 2, "int8": 1}


def flops_full_frame_g(tile_g: float) -> float:
    """Scale a 256-tile GFLOP count to the 1024x1224 band plane.

    Args:
        tile_g: Locked 256-tile G count.

    Returns:
        Full-frame G count.
    """
    return tile_g * area_scale()


def cls_flops_ff_g() -> float:
    """Return classifier full-frame GFLOPs."""
    return flops_full_frame_g(CLS_FLOPS_TILE_G)


def seg_flops_ff_g() -> float:
    """Return segmentor full-frame GFLOPs."""
    return flops_full_frame_g(SEG_FLOPS_TILE_G)


def input_bytes() -> float:
    """Return NCHW float32 input tensor bytes (graph I/O stays float32)."""
    return float(IN_CHANNELS * FRAME_H_PX * FRAME_W_PX * 4)


def activation_bytes() -> float:
    """Return live feature-map bytes at output stride 4, float32."""
    height = FRAME_H_PX // OUTPUT_STRIDE
    width = FRAME_W_PX // OUTPUT_STRIDE
    return float(LIVE_MAPS * LIVE_CHANNELS * height * width * 4)


def weight_bytes(kind: str, precision: str) -> float:
    """Return parameter bytes at ``precision``.

    Args:
        kind: ``cls`` or ``seg``.
        precision: ``fp32``, ``fp16``, or ``int8``.

    Returns:
        Weight footprint in bytes.

    Raises:
        ValueError: If ``kind`` or ``precision`` is unknown.
    """
    if kind == "cls":
        params = CLS_PARAMS
    elif kind == "seg":
        params = SEG_PARAMS
    else:
        raise ValueError(f"unknown kind {kind!r}")
    elem = _ELEM_BYTES.get(precision)
    if elem is None:
        raise ValueError(f"unknown precision {precision!r}")
    return float(params * elem)


def bytes_moved(kind: str, precision: str) -> float:
    """Return estimated bytes moved per forward pass.

    Args:
        kind: ``cls`` or ``seg``.
        precision: ``fp32``, ``fp16``, or ``int8``.

    Returns:
        Sum of input, weights, and live maps.
    """
    return input_bytes() + weight_bytes(kind, precision) + activation_bytes()
