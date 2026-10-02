"""Shape contract for GSD-conditioned ONNX model graphs.

The flight graph has two float inputs, ``image`` (N, C, H, W) and ``gsd``
(N, 2), with classifier or segmentor logits as its single output. Batch is
always dynamic; configured flight consumers also require the exact tile size.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

from flight.libs.types import Err, FaultCode, Ok, Result

Shape = tuple[int | None, ...]


def verify_conditioned_shapes(
    image: Shape,
    gsd: Shape,
    output: Shape,
    channels: int,
    tile_hw: tuple[int, int] | None,
    kind: str,
) -> Result[None, FaultCode]:
    """Validate dynamic-batch image/GSD inputs and classifier/segmentor output.

    Spatial dimensions must equal ``tile_hw`` when supplied. Without a tile size,
    they may be dynamic, while any concrete image and segmentor dimensions agree.
    """
    valid_tile = tile_hw is None or (
        isinstance(tile_hw, tuple)
        and len(tile_hw) == 2
        and all(type(value) is int and value > 0 for value in tile_hw)
    )
    valid_image = (
        len(image) == 4
        and image[0] is None
        and type(image[1]) is int
        and image[1] == channels
        and all(value is None or (type(value) is int and value > 0) for value in image[2:])
    )
    valid_gsd = len(gsd) == 2 and gsd[0] is None and type(gsd[1]) is int and gsd[1] == 2
    valid = type(channels) is int and channels > 0 and valid_image and valid_gsd and valid_tile
    if not valid:
        return Err(FaultCode.MODEL_CORRUPT)
    height, width = image[2], image[3]
    if tile_hw is not None and (height, width) != tile_hw:
        return Err(FaultCode.MODEL_CORRUPT)
    if kind == "classifier":
        valid_output = (
            len(output) == 2 and output[0] is None and type(output[1]) is int and output[1] == 1
        )
        return Ok(None) if valid_output else Err(FaultCode.MODEL_CORRUPT)
    if kind == "segmentor":
        if (
            len(output) != 4
            or output[:2] != (None, 1)
            or type(output[1]) is not int
            or any(
                value is not None and (type(value) is not int or value <= 0) for value in output[2:]
            )
        ):
            return Err(FaultCode.MODEL_CORRUPT)
        out_hw = output[2:]
        if tile_hw is not None:
            return Ok(None) if out_hw == tile_hw else Err(FaultCode.MODEL_CORRUPT)
        if any(
            i is not None and o is not None and i != o
            for i, o in zip((height, width), out_hw, strict=True)
        ):
            return Err(FaultCode.MODEL_CORRUPT)
        return Ok(None)
    return Err(FaultCode.MODEL_CORRUPT)
