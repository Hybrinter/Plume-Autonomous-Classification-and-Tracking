"""Shape contract for GSD-conditioned ONNX model graphs.

Pure, deterministic verification of the two-input flight graph contract:
``image`` (N, C, H, W) + ``gsd`` (N, 2) in, conditioned logits out. Both
tools-side export/session validation and the flight loader reuse this single
implementation of the shape formulas; None entries are dynamic dimensions.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

# internal
from flight.libs.types import Err, FaultCode, Ok, Result

Shape = tuple[int | None, ...]


def verify_conditioned_shapes(
    image: Shape,
    gsd: Shape,
    output: Shape,
    channels: int,
    tile_hw: tuple[int, int],
    kind: str,
) -> Result[None, FaultCode]:
    """Check GSD-conditioned graph shapes for the given model family.

    Args:
        image: Model image input shape, must be (None, C, H, W); None entries
            mark dynamic dims, spatial dims may be None or the tile size.
        gsd: Encoded GSD input shape, must be (None, 2).
        output: Model output shape. The classifier accepts (None, 1). The
            segmentor accepts (None, 1, H, W) with spatial dims dynamic or
            equal to tile_hw.
        channels: Required image channel count.
        tile_hw: Flight trace tile (height, width) in pixels.
        kind: Model kind ("classifier" or "segmentor").

    Returns:
        Ok(None) on conformance, else Err(FaultCode.MODEL_CORRUPT).

    Notes:
        The batch dim must be dynamic (None) on every I/O tensor.
    """
    height, width = tile_hw
    if (
        len(image) != 4
        or image[0] is not None
        or image[1] != channels
        or image[2] not in (None, height)
        or image[3] not in (None, width)
        or gsd != (None, 2)
    ):
        return Err(FaultCode.MODEL_CORRUPT)
    if kind == "classifier":
        if output != (None, 1):
            return Err(FaultCode.MODEL_CORRUPT)
        return Ok(None)
    if kind == "segmentor":
        if (
            len(output) != 4
            or output[0] is not None
            or output[1] != 1
            or output[2] not in (None, height)
            or output[3] not in (None, width)
        ):
            return Err(FaultCode.MODEL_CORRUPT)
        return Ok(None)
    return Err(FaultCode.MODEL_CORRUPT)
