"""Schema-2 unit-tile product: shared layout and capture validation.

A unit tile is one float32 ``(C, H, W)`` image holding unit-domain pixels,
plus the capture metadata a ground importer needs for provenance and GSD
conditioning. These types and validators define the shared contract; file
I/O, label authority, and storage policy stay outside this module.

Contains:
  - UNIT_TILE_SOURCE_SCHEMA: the on-disk schema identifier.
  - UnitTileLayout: recorded bands, tile size, grid, and GSD reference.
  - UnitTileCapture: per-tile provenance; no label or ground-truth mask.
  - validate_layout / validate_capture / validate_unit_tile: pure validators
    returning ``Err(FRAME_MALFORMED)`` on invalid data.

Satisfies: REQ-AIML-PREP-002.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass

import numpy as np

from flight.libs.types import Err, FaultCode, Ok, Result
from flight.payload.gimbal.footprint import GsdPair

UNIT_TILE_SOURCE_SCHEMA = 2


@dataclass(frozen=True, slots=True)
class UnitTileLayout:
    """Recorded layout of one unit-tile product.

    Attributes:
        band_names: Nonempty unique channel names in storage order.
        tile_hw: ``(height, width)`` in pixels.
        grid: ``(rows, cols)`` of the frame grid the tiles came from.
        gsd_reference_m: Reference metres for model-GSD encoding.
    """

    band_names: tuple[str, ...]
    tile_hw: tuple[int, int]
    grid: tuple[int, int]
    gsd_reference_m: float


@dataclass(frozen=True, slots=True)
class UnitTileCapture:
    """Provenance for one stored unit tile.

    Attributes:
        tile_id: Safe file stem.
        frame_id: Frame that contained the tile.
        grid_rc: ``(row, col)`` inside ``layout.grid``.
        theta_g_deg: Gimbal elevation at the shutter, in degrees.
        gsd: Pixel GSD at the tile center, lateral then along-track metres.
        gsd_nominal: True when ``gsd`` is nominal orbit geometry rather than
            measured capture geometry.
    """

    tile_id: str
    frame_id: str
    grid_rc: tuple[int, int]
    theta_g_deg: float
    gsd: GsdPair
    gsd_nominal: bool = False


def validate_layout(layout: object) -> Result[None, FaultCode]:
    """Return ``Ok(None)`` when ``layout`` is a well-formed unit-tile layout.

    Args:
        layout: Candidate ``UnitTileLayout``.

    Returns:
        Result[None, FaultCode]: ``Ok(None)`` when valid, else
            ``Err(FRAME_MALFORMED)``.
    """
    if not isinstance(layout, UnitTileLayout):
        return Err(FaultCode.FRAME_MALFORMED)
    bands = layout.band_names
    if (
        not isinstance(bands, tuple)
        or len(bands) < 1
        or any(not isinstance(name, str) or not name for name in bands)
        or len(set(bands)) != len(bands)
    ):
        return Err(FaultCode.FRAME_MALFORMED)
    for pair in (layout.tile_hw, layout.grid):
        if (
            not isinstance(pair, tuple)
            or len(pair) != 2
            or any(not isinstance(dim, int) or isinstance(dim, bool) for dim in pair)
            or min(pair) < 1
        ):
            return Err(FaultCode.FRAME_MALFORMED)
    reference = _finite_float(layout.gsd_reference_m)
    if reference is None or reference <= 0.0:
        return Err(FaultCode.FRAME_MALFORMED)
    return Ok(None)


def validate_capture(capture: object, layout: object) -> Result[None, FaultCode]:
    """Return ``Ok(None)`` when ``capture`` is valid under ``layout``.

    Args:
        capture: Candidate ``UnitTileCapture``.
        layout: Candidate ``UnitTileLayout``; an invalid layout rejects.

    Returns:
        Result[None, FaultCode]: ``Ok(None)`` when valid, else
            ``Err(FRAME_MALFORMED)``.
    """
    if not isinstance(layout, UnitTileLayout) or isinstance(validate_layout(layout), Err):
        return Err(FaultCode.FRAME_MALFORMED)
    if not isinstance(capture, UnitTileCapture):
        return Err(FaultCode.FRAME_MALFORMED)
    if not _safe_id(capture.tile_id):
        return Err(FaultCode.FRAME_MALFORMED)
    if not isinstance(capture.frame_id, str) or not capture.frame_id:
        return Err(FaultCode.FRAME_MALFORMED)
    grid_rc = capture.grid_rc
    rows, cols = layout.grid
    if (
        not isinstance(grid_rc, tuple)
        or len(grid_rc) != 2
        or any(not isinstance(index, int) or isinstance(index, bool) for index in grid_rc)
        or grid_rc[0] < 0
        or grid_rc[0] >= rows
        or grid_rc[1] < 0
        or grid_rc[1] >= cols
    ):
        return Err(FaultCode.FRAME_MALFORMED)
    if _finite_float(capture.theta_g_deg) is None:
        return Err(FaultCode.FRAME_MALFORMED)
    gsd = capture.gsd
    if not isinstance(gsd, GsdPair):
        return Err(FaultCode.FRAME_MALFORMED)
    lateral = _finite_float(gsd.lateral_m)
    along = _finite_float(gsd.along_m)
    if lateral is None or along is None or lateral <= 0.0 or along <= 0.0:
        return Err(FaultCode.FRAME_MALFORMED)
    if not isinstance(capture.gsd_nominal, bool):
        return Err(FaultCode.FRAME_MALFORMED)
    return Ok(None)


def validate_unit_tile(image: object, layout: object) -> Result[None, FaultCode]:
    """Return ``Ok(None)`` when ``image`` is a valid unit tile under ``layout``.

    The array must already be float32 ``(len(band_names), H, W)`` with finite
    pixels inside ``[0, 1]``; no casting, clipping, or mutation is performed.

    Args:
        image: Candidate ``np.ndarray``.
        layout: Candidate ``UnitTileLayout``; an invalid layout rejects.

    Returns:
        Result[None, FaultCode]: ``Ok(None)`` when valid, else
            ``Err(FRAME_MALFORMED)``.
    """
    if not isinstance(layout, UnitTileLayout) or isinstance(validate_layout(layout), Err):
        return Err(FaultCode.FRAME_MALFORMED)
    expected = (len(layout.band_names), layout.tile_hw[0], layout.tile_hw[1])
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.float32
        or image.shape != expected
        or not np.all(np.isfinite(image))
        or not np.all((image >= 0.0) & (image <= 1.0))
    ):
        return Err(FaultCode.FRAME_MALFORMED)
    return Ok(None)


def _finite_float(value: object) -> float | None:
    """Return ``float(value)`` for a finite real number, else ``None``."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    try:
        converted = float(value)
    except TypeError, ValueError, OverflowError:
        return None
    return converted if math.isfinite(converted) else None


def _safe_id(value: object) -> bool:
    """Return True for a nonempty file-stem identifier."""
    return (
        isinstance(value, str)
        and bool(value)
        and value not in (".", "..")
        and "/" not in value
        and "\\" not in value
        and "\x00" not in value
    )
