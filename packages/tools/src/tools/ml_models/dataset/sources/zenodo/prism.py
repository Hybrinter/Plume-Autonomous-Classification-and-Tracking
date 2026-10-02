"""AP-3200T prism weights and the Sentinel-2 to color mix.

Contains:
  - WeightTable, load_weight_table: per-color Sentinel-2 weights.
  - mix_prism: L2A counts to BLUE, GREEN, RED reflectance.

The committed table is curve-height readings of the AP-3200T solid IR-cut
figure. It is not a laboratory integral. B5 and longer bands are absent.
Their weight is 0. This module does not download archives.
"""

from __future__ import annotations

import math
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

_DN_SCALE = np.float32(10000.0)
_COLOR_NAMES: tuple[str, ...] = ("blue", "green", "red")
_TABLE_KEYS: frozenset[str] = frozenset({"id", "blue", "green", "red"})


@dataclass(frozen=True, slots=True)
class WeightTable:
    """Per-color Sentinel-2 weights.

    Attributes:
        id: Table identifier stored on the dataset manifest.
        blue: Band id to weight. The weights sum to 1.
        green: Band id to weight. The weights sum to 1.
        red: Band id to weight. The weights sum to 1.
    """

    id: str
    blue: Mapping[str, float]
    green: Mapping[str, float]
    red: Mapping[str, float]

    def color_map(self, name: str) -> Mapping[str, float]:
        """Return the weight map for ``blue``, ``green``, or ``red``.

        Args:
            name: Color name.

        Returns:
            Mapping[str, float]: Band weights for that color.

        Raises:
            ValueError: If ``name`` is not a color.
        """
        if name == "blue":
            return self.blue
        if name == "green":
            return self.green
        if name == "red":
            return self.red
        raise ValueError(f"unknown color {name!r}")


def load_weight_table(path: str | Path) -> WeightTable:
    """Load a prism weight table.

    Args:
        path: TOML file with ``id`` and ``[blue]``, ``[green]``, and ``[red]``
            tables.

    Returns:
        WeightTable: Identifier and one immutable map per color.

    Raises:
        FileNotFoundError: If ``path`` is missing.
        ValueError: If a color is missing or its weights do not sum to 1
            within ``1e-6``.
        tomllib.TOMLDecodeError: If the file is not TOML.
    """
    dest = Path(path)
    if not dest.is_file():
        raise FileNotFoundError(dest)
    raw = tomllib.loads(dest.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("weight table must be a TOML table")
    extra = sorted(set(raw) - _TABLE_KEYS)
    missing = sorted(_TABLE_KEYS - set(raw))
    if extra or missing:
        raise ValueError(f"weight table keys mismatch; missing={missing} extra={extra}")
    table_id = raw["id"]
    if not isinstance(table_id, str) or not table_id:
        raise ValueError("weight table id must be a non-empty string")
    return WeightTable(
        id=table_id,
        blue=MappingProxyType(_color_weights(raw, "blue")),
        green=MappingProxyType(_color_weights(raw, "green")),
        red=MappingProxyType(_color_weights(raw, "red")),
    )


def mix_prism(
    stack_dn: np.ndarray,
    band_ids: Sequence[str],
    table: WeightTable,
) -> np.ndarray:
    """Mix Sentinel-2 L2A counts into BLUE, GREEN, RED reflectance.

    Args:
        stack_dn: Array ``(C, H, W)`` of L2A digital numbers.
        band_ids: Sentinel-2 id of each plane, length ``C``.
        table: Per-color weights. A band absent from a color contributes 0.

    Returns:
        np.ndarray[float32, (3, H, W)]: Planes in order BLUE, GREEN, RED.
        Each plane is the weighted sum of ``clip(stack_dn / 10000, 0, 1)``.

    Raises:
        ValueError: If the stack rank disagrees with ``band_ids``, a band id
            repeats, or the table names a band id that is not in ``band_ids``.
    """
    stack = np.asarray(stack_dn)
    if stack.ndim != 3:
        raise ValueError(f"stack_dn must have shape (C, H, W); got {stack.shape}")
    ids = tuple(band_ids)
    if int(stack.shape[0]) != len(ids):
        raise ValueError(f"stack channels {stack.shape[0]} != len(band_ids) {len(ids)}")
    if int(stack.shape[1]) < 1 or int(stack.shape[2]) < 1:
        raise ValueError(f"stack spatial shape must be positive; got {stack.shape}")
    index_by_id: dict[str, int] = {}
    for index, band_id in enumerate(ids):
        if band_id in index_by_id:
            raise ValueError(f"duplicate band id {band_id}")
        index_by_id[band_id] = index
    for color in _COLOR_NAMES:
        for band_id in table.color_map(color):
            if band_id not in index_by_id:
                raise ValueError(f"unknown band id {band_id}")
    reflectance = np.clip(
        np.asarray(stack, dtype=np.float32) / _DN_SCALE,
        0.0,
        1.0,
    )  # np.ndarray[float32, (C, H, W)]
    mixed = np.zeros((3, int(stack.shape[1]), int(stack.shape[2])), dtype=np.float32)
    for plane, color in enumerate(_COLOR_NAMES):
        for band_id, weight in table.color_map(color).items():
            mixed[plane] += np.float32(weight) * reflectance[index_by_id[band_id]]
    return mixed


def _color_weights(raw: dict[str, object], color: str) -> dict[str, float]:
    """Return finite weights for one color, summing to 1.

    Args:
        raw: Decoded TOML object.
        color: ``blue``, ``green``, or ``red``.

    Returns:
        dict[str, float]: Band id to weight.

    Raises:
        ValueError: If the table is missing, a weight is not finite, or the
            sum is not 1 within ``1e-6``.
    """
    value = raw[color]
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{color} must be a non-empty table of band weights")
    weights: dict[str, float] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{color} band id must be a string")
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{color} weight for {key} must be a number")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{color} weight for {key} must be finite")
        weights[key] = number
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"{color} weights sum to {total}; expected 1 within 1e-6")
    return weights
