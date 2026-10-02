"""Offline dihedral augmentation for finished dataset rows.

Contains:
  - ELEMENT_NAMES, SHAPE_PRESERVING: the eight dihedral names and the four
    that keep H and W.
  - AugmentRecipe: the element list requested by a build.
  - legal_elements: eight names on a square tile, otherwise the four
    shape-preserving names.
  - apply_dihedral: one element applied to a ``(C, H, W)`` array.

GSD is not an input. A 90 degree or transpose element swaps H and W, so a
non-square tile drops those elements and keeps along-track compression on H.
"""

from __future__ import annotations

from typing import Self

import numpy as np
from pydantic import ConfigDict, model_validator
from pydantic.dataclasses import dataclass as pydantic_dataclass

ELEMENT_NAMES: tuple[str, ...] = (
    "id",
    "rot90",
    "rot180",
    "rot270",
    "flip_h",
    "flip_v",
    "transpose",
    "anti_transpose",
)
SHAPE_PRESERVING: tuple[str, ...] = ("id", "rot180", "flip_h", "flip_v")
_KNOWN: frozenset[str] = frozenset(ELEMENT_NAMES)
_SCHEMA = ConfigDict(extra="forbid")
# k quarter-turns after an optional left-right flip. Odd k swaps H and W.
_ELEMENT_OP: dict[str, tuple[int, bool]] = {
    "id": (0, False),
    "rot90": (1, False),
    "rot180": (2, False),
    "rot270": (3, False),
    "flip_h": (0, True),
    "flip_v": (2, True),
    "transpose": (1, True),
    "anti_transpose": (3, True),
}


@pydantic_dataclass(frozen=True, slots=True, config=_SCHEMA)
class AugmentRecipe:
    """Element names requested for train rows.

    Attributes:
        elements: Subset of ``ELEMENT_NAMES`` in application order. Duplicates
            are rejected.
    """

    elements: tuple[str, ...] = ELEMENT_NAMES

    @model_validator(mode="after")
    def _known_elements(self) -> Self:
        """Reject an empty list, an unknown name, or a duplicate."""
        if len(self.elements) < 1:
            raise ValueError("augment elements must be non-empty")
        unknown = [name for name in self.elements if name not in _KNOWN]
        if unknown:
            raise ValueError(f"unknown augment elements {unknown}")
        if len(set(self.elements)) != len(self.elements):
            raise ValueError("augment elements must be unique")
        return self


def legal_elements(height: int, width: int) -> tuple[str, ...]:
    """Return the dihedral elements that preserve the along-track axis.

    Args:
        height: Tile H in pixels.
        width: Tile W in pixels.

    Returns:
        tuple[str, ...]: All eight names when ``height == width``. Otherwise
        ``id``, ``rot180``, ``flip_h``, and ``flip_v``.

    Raises:
        ValueError: If ``height`` or ``width`` is below 1.
    """
    if height < 1 or width < 1:
        raise ValueError(f"tile size must be >= 1; got {height}x{width}")
    if height == width:
        return ELEMENT_NAMES
    return SHAPE_PRESERVING


def apply_dihedral(image: np.ndarray, element: str) -> np.ndarray:
    """Apply one dihedral element to a channel-first array.

    Args:
        image: np.ndarray[(C, H, W)].
        element: One name from ``ELEMENT_NAMES``.

    Returns:
        np.ndarray[(C, H', W')]: Contiguous copy. ``H'`` and ``W'`` swap when
        the element is a 90 degree turn or a transpose.

    Raises:
        ValueError: If ``element`` is unknown or the array is not 3-D.
    """
    if image.ndim != 3:
        raise ValueError(f"dihedral input must be (C, H, W); got {image.shape}")
    op = _ELEMENT_OP.get(element)
    if op is None:
        raise ValueError(f"unknown augment element {element!r}")
    turns, mirror = op
    out = np.flip(image, axis=2) if mirror else image
    if turns:
        out = np.rot90(out, turns, axes=(1, 2))
    return np.ascontiguousarray(out)
