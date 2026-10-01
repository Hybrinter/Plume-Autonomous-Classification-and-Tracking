"""Schema-2 model sidecar for a GSD-conditioned ONNX artifact.

Contains:
  - ModelManifest: frozen, extra-forbidden metadata contract.
  - sidecar_path: the sibling ``.json`` path for an artifact.
  - write_manifest / load_manifest: JSON serialization, no pickle.
"""

from __future__ import annotations

import json
import math
from dataclasses import field
from pathlib import Path
from typing import Literal, Self

from flight.libs.types import Err
from pydantic import ConfigDict, TypeAdapter, model_validator
from pydantic.dataclasses import dataclass as pydantic_dataclass

from tools.ml_models.export.contract import (
    CONDITIONING_ID,
    GSD_ENCODING,
    verify_conditioned_shapes,
)

SCHEMA_VERSION = 2
BAND_NAMES = ("BLUE", "GREEN", "RED")
TILE_HW = (193, 258)
GRID = (8, 8)
FRAME_HW = (1544, 2064)
COVERAGE_ALTITUDE_M = 460_000.0
_QUANTIZATION = "fp32"
_SCHEMA = ConfigDict(extra="forbid")
_HEX = frozenset("0123456789abcdefABCDEF")
_FAMILY_PREFIX = {"classifier": "pactnet", "segmentor": "dilatenet"}


def _is_hex64(value: str) -> bool:
    """Return True when ``value`` is exactly 64 hexadecimal characters."""
    return len(value) == 64 and all(char in _HEX for char in value)


@pydantic_dataclass(frozen=True, slots=True, config=_SCHEMA)
class ModelManifest:
    """Validated sidecar describing one exported conditioned ONNX artifact.

    Attributes:
        version: Manifest version string (first 16 hex of the artifact hash).
        kind: ``classifier`` or ``segmentor``.
        arch: Architecture name; prefix must match the kind family.
        sha256: SHA-256 digest of the artifact bytes.
        dataset_hash: Combined training dataset hash.
        band_names: Model input bands, exactly ``("BLUE", "GREEN", "RED")``.
        input_shape: Declared image input shape; None entries are dynamic.
        gsd_input_shape: Declared GSD input shape, always ``(None, 2)``.
        output_shape: Declared logits output shape; None entries are dynamic.
        gsd_reference_m: Reference GSD used by the conditioning encoding.
        gsd_min_m: Actual minimum training GSD as (lateral, along).
        gsd_max_m: Actual maximum training GSD as (lateral, along).
        conditioning: Conditioning marker, always ``film-log-gsd-v1``.
        gsd_encoding: Encoding marker, always
            ``ln_metres_over_reference_lateral_along``.
        input_types: Validated I/O dtypes, ``{"image": "float32",
            "gsd": "float32"}``.
        output_type: Validated output dtype, ``"float32"``.
        norm: Preprocessing normalization marker, always ``"unit"``.
        schema: Sidecar schema version, always 2.
        quantization: Artifact quantization, ``fp32`` for this schema.
        tile_hw: Flight trace tile (height, width).
        grid: Tile grid (rows, cols) used for coverage checks.
        frame_hw: Full sensor frame (height, width).
        coverage_altitude_m: Altitude used for required coverage.
        partial_gsd: True when actual coverage does not span required coverage.
    """

    version: str
    kind: Literal["classifier", "segmentor"]
    arch: str
    sha256: str
    dataset_hash: str
    band_names: tuple[str, ...]
    input_shape: tuple[int | None, ...]
    gsd_input_shape: tuple[int | None, ...]
    output_shape: tuple[int | None, ...]
    gsd_reference_m: float
    gsd_min_m: tuple[float, float]
    gsd_max_m: tuple[float, float]
    conditioning: str
    gsd_encoding: str
    input_types: dict[str, str] = field(
        default_factory=lambda: {"image": "float32", "gsd": "float32"}
    )
    output_type: str = "float32"
    norm: str = "unit"
    schema: int = SCHEMA_VERSION
    quantization: str = _QUANTIZATION
    tile_hw: tuple[int, int] = TILE_HW
    grid: tuple[int, int] = GRID
    frame_hw: tuple[int, int] = FRAME_HW
    coverage_altitude_m: float = COVERAGE_ALTITUDE_M
    partial_gsd: bool = False

    @model_validator(mode="after")
    def _contract(self) -> Self:
        """Reject anything outside the conditioned flight export contract."""
        if self.schema != SCHEMA_VERSION:
            raise ValueError(f"schema must be {SCHEMA_VERSION}; got {self.schema}")
        if self.arch.split("_")[0] != _FAMILY_PREFIX[self.kind]:
            raise ValueError(
                f"{self.kind} arch must start with {_FAMILY_PREFIX[self.kind]!r}; got {self.arch!r}"
            )
        if self.conditioning != CONDITIONING_ID:
            raise ValueError(f"conditioning must be {CONDITIONING_ID!r}")
        if self.gsd_encoding != GSD_ENCODING:
            raise ValueError(f"gsd_encoding must be {GSD_ENCODING!r}")
        if dict(self.input_types) != {"image": "float32", "gsd": "float32"}:
            raise ValueError("input_types must be {'image': 'float32', 'gsd': 'float32'}")
        if self.output_type != "float32":
            raise ValueError("output_type must be 'float32'")
        if self.norm != "unit":
            raise ValueError("norm must be 'unit'")
        if tuple(self.band_names) != BAND_NAMES:
            raise ValueError(f"band_names must be {BAND_NAMES}; got {self.band_names}")
        if not math.isfinite(self.gsd_reference_m) or self.gsd_reference_m <= 0.0:
            raise ValueError("gsd_reference_m must be finite and > 0")
        for name, values in (("gsd_min_m", self.gsd_min_m), ("gsd_max_m", self.gsd_max_m)):
            if not all(math.isfinite(v) and v > 0.0 for v in values):
                raise ValueError(f"{name} entries must be finite and > 0")
        if any(low > high for low, high in zip(self.gsd_min_m, self.gsd_max_m, strict=True)):
            raise ValueError("gsd_min_m must be <= gsd_max_m per axis")
        if not _is_hex64(self.sha256):
            raise ValueError("sha256 must be 64 hex characters")
        if not _is_hex64(self.dataset_hash):
            raise ValueError("dataset_hash must be 64 hex characters")
        if self.tile_hw != TILE_HW:
            raise ValueError(f"tile_hw must be {TILE_HW}; got {self.tile_hw}")
        if self.grid != GRID:
            raise ValueError(f"grid must be {GRID}; got {self.grid}")
        if self.frame_hw != FRAME_HW:
            raise ValueError(f"frame_hw must be {FRAME_HW}; got {self.frame_hw}")
        if not math.isfinite(self.coverage_altitude_m) or self.coverage_altitude_m <= 0.0:
            raise ValueError("coverage_altitude_m must be finite and > 0")
        result = verify_conditioned_shapes(
            self.input_shape,
            self.gsd_input_shape,
            self.output_shape,
            len(BAND_NAMES),
            self.tile_hw,
            self.kind,
        )
        if isinstance(result, Err):
            raise ValueError(f"declared shapes violate the flight contract ({result.error.value})")
        return self


def sidecar_path(artifact: str | Path) -> Path:
    """Return the sibling sidecar path for an ONNX artifact."""
    return Path(artifact).with_suffix(".json")


def acceptance_path(artifact: str | Path) -> Path:
    """Return the sibling acceptance-report path for an ONNX artifact."""
    return Path(artifact).with_suffix(".acceptance.json")


def write_manifest(path: str | Path, manifest: ModelManifest) -> None:
    """Serialize ``manifest`` to ``path`` as JSON."""
    adapter = TypeAdapter(ModelManifest)
    payload = adapter.dump_python(manifest, mode="json")
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_manifest(path: str | Path) -> ModelManifest:
    """Parse and validate a model sidecar.

    Raises:
        OSError / json.JSONDecodeError: On a missing or malformed file.
        ValueError: If the payload fails the schema or contract checks.
    """
    dest = Path(path)
    raw = json.loads(dest.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{dest.name} must be a JSON object")
    return TypeAdapter(ModelManifest).validate_python(raw)
