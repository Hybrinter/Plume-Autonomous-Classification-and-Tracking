"""Raw tile contract consumed by the dataset build.

Contains:
  - GsdPair: re-export of ``flight.payload.gimbal.footprint.GsdPair``, so a
    tile carries the same metres-per-pixel pair flight computes.
  - ConditionTag, ObservationMetadata, PreparedMaskState: recorded source
    provenance for one original observation.
  - prepared_mask_state: classify the actual prepared source mask.
  - BinSpec: an optional GSD bin row.
  - RawTileRef, RawTile: one indexed row and its arrays.
  - Domain: ``unit``.
  - RawSource: protocol for a forward-only tile stream.

``tile_hw`` is a property of each row. A source may emit more than one spatial
size. ``iter_tiles`` yields rows in ``index`` order in one pass.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from flight.payload.gimbal.footprint import GsdPair
from pydantic import ConfigDict, StrictStr, model_validator
from pydantic.dataclasses import dataclass as schema_dataclass

Domain = str

__all__ = [
    "BinSpec",
    "ConditionTag",
    "Domain",
    "GsdPair",
    "ObservationMetadata",
    "RawSource",
    "RawTile",
    "RawTileRef",
    "prepared_mask_state",
]

type SourceAnnotationState = Literal["UNKNOWN", "MISSING", "EXPLICIT_EMPTY", "NONEMPTY"]
type PreparedMaskState = Literal["UNKNOWN", "MISSING", "EMPTY", "NONEMPTY"]
_METADATA_SCHEMA = ConfigDict(extra="forbid")
_UTC_TIMESTAMP = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]"
    r"(?:\.[0-9]{1,6})?Z"
)


@schema_dataclass(frozen=True, slots=True, config=_METADATA_SCHEMA)
class ConditionTag:
    """One explicitly recorded categorical condition, never inferred from pixels."""

    name: StrictStr
    value: StrictStr

    @model_validator(mode="after")
    def _bounds(self) -> ConditionTag:
        """Reject blank names and values rather than manufacturing a category."""
        if not self.name.strip() or not self.value.strip():
            raise ValueError("condition tag name and value must be nonblank")
        return self


@schema_dataclass(frozen=True, slots=True, config=_METADATA_SCHEMA)
class ObservationMetadata:
    """Authoritative source identity, UTC time, conditions and annotation provenance.

    An observation ID identifies one original source tile before GSD variants,
    task copies and augmentation; it does not assert statistical independence.
    UNKNOWN source annotation state is distinct from missing annotations.
    NONEMPTY means source annotation entries exist, not that rasterized plume
    pixels or physical annotation area are nonempty. Original polygon geometry
    can be absent for an imported prepared mask; that stays UNKNOWN.
    """

    observation_id: StrictStr | None = None
    acquired_at_utc: StrictStr | None = None
    conditions: tuple[ConditionTag, ...] = field(default=())
    annotation_source: StrictStr | None = None
    annotation_version: StrictStr | None = None
    source_annotation_state: SourceAnnotationState = "UNKNOWN"

    @model_validator(mode="after")
    def _bounds(self) -> ObservationMetadata:
        """Require explicit valid UTC time and uniquely named recorded tags."""
        for value in (self.observation_id, self.annotation_source, self.annotation_version):
            if value is not None and not value.strip():
                raise ValueError("recorded metadata strings must be nonblank or null")
        if self.acquired_at_utc is not None:
            if _UTC_TIMESTAMP.fullmatch(self.acquired_at_utc) is None:
                raise ValueError("acquired_at_utc must be an ISO UTC timestamp ending in Z")
            datetime.fromisoformat(self.acquired_at_utc)
        if len({tag.name for tag in self.conditions}) != len(self.conditions):
            raise ValueError("condition names must be unique")
        return self


def prepared_mask_state(mask: np.ndarray | None) -> PreparedMaskState:
    """Classify the actual prepared binary source mask without using its class label.

    The build validates binary mask geometry before calling this helper. This
    state is shared across task copies, including classifier copies that do
    not store a mask; it never implies that a missing mask is verified empty.
    """
    return "MISSING" if mask is None else "NONEMPTY" if bool(np.any(mask)) else "EMPTY"


@dataclass(frozen=True, slots=True)
class BinSpec:
    """One named GSD bin recorded on the finished dataset.

    Attributes:
        bin_id: Stable bin name.
        lateral_m: Nominal lateral GSD in metres.
        along_m: Nominal along-track GSD in metres.
        elevation_deg: Gimbal elevation for a flight bin. None for a native bin.
    """

    bin_id: str
    lateral_m: float
    along_m: float
    elevation_deg: float | None = None


@dataclass(frozen=True, slots=True)
class RawTileRef:
    """Identity and geometry of one raw tile, without pixel arrays.

    Attributes:
        tile_id: Unique id inside the source.
        group_id: Split key. Rows that share a group stay in one split.
        label: Classification target. Values at or above 0.5 count as positive.
        has_mask: True when the segmentor set includes this row.
        gsd: Pixel GSD at the tile center.
        height: Image height in pixels.
        width: Image width in pixels.
        frame_id: Flight frame id. None when the source has no frame.
        grid_rc: ``(row, col)`` on the flight grid. None when absent.
        bin_id: GSD bin name. Empty when the source has a single geometry.
        theta_g_deg: Gimbal elevation at the shutter, in degrees. None when
            the source does not record one.
        gsd_nominal: True when ``gsd`` comes from nominal orbit geometry
            rather than a measured capture.
        metadata: Recorded provenance of the original source observation.
            Defaults to all-unknown; the build copies it onto every stored
            row derived from this tile.
    """

    tile_id: str
    group_id: str
    label: float
    has_mask: bool
    gsd: GsdPair
    height: int
    width: int
    frame_id: str | None
    grid_rc: tuple[int, int] | None
    bin_id: str
    theta_g_deg: float | None = None
    gsd_nominal: bool = False
    metadata: ObservationMetadata = field(default_factory=ObservationMetadata)


@dataclass(frozen=True, slots=True)
class RawTile:
    """One raw tile yielded by a source.

    Attributes:
        ref: Row identity.
        image: np.ndarray[float32, (C, H, W)] in the unit interval, where
            ``(H, W)`` is ``(ref.height, ref.width)``.
        mask: np.ndarray[(1, H, W)] or None. Present exactly when ``has_mask``.
    """

    ref: RawTileRef
    image: np.ndarray
    mask: np.ndarray | None


@runtime_checkable
class RawSource(Protocol):
    """Forward-only producer of raw tiles.

    Attributes:
        name: Short source name stored on the manifest.
        band_names: Channel names in image order.
        domain: ``unit``. Prepared tiles are already float32 unit pixels.
        source_ref: DOI or other provenance string. Empty when the source
            has no external origin.
        bins: Bin table copied onto the manifest. Empty when the source has
            no named bins.
    """

    name: str
    band_names: tuple[str, ...]
    domain: str
    source_ref: str
    bins: tuple[BinSpec, ...]

    def index(self) -> tuple[RawTileRef, ...]:
        """Return every row in stream order.

        Returns:
            tuple[RawTileRef, ...]: One ref per tile, same order as ``iter_tiles``.
        """
        ...

    def iter_tiles(self) -> Iterator[RawTile]:
        """Yield each tile once, in ``index`` order.

        Returns:
            Iterator[RawTile]: One forward pass.
        """
        ...
