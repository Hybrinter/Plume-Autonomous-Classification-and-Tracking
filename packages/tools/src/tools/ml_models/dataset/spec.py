"""Build specification for a finished dataset.

Contains:
  - BuildSpec: split recipe, augment recipe, tasks, bands, and GSD reference.
  - load_build_spec: TOML reader.
  - TASK_NAMES: ``classifier`` and ``segmentor``.

The same spec is applied to every source. Val and test rows are not augmented.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Self

from flight.payload.gimbal.footprint import GSD_REFERENCE_M
from pydantic import ConfigDict, model_validator
from pydantic.dataclasses import dataclass as pydantic_dataclass

from tools.ml_models.dataset.augment import AugmentRecipe
from tools.ml_models.dataset.split import SplitRecipe

TASK_NAMES: tuple[str, ...] = ("classifier", "segmentor")
_SCHEMA = ConfigDict(extra="forbid")


@pydantic_dataclass(frozen=True, slots=True, config=_SCHEMA)
class BuildSpec:
    """Controls split assignment, augmentation, and which tasks are written.

    Attributes:
        split: Group split recipe shared by every task and every bin.
        augment: Train-only dihedral elements. Intersected with the elements
            legal for each tile's height and width.
        tasks: Non-empty subset of ``classifier`` and ``segmentor``.
        input_bands: Optional expected channel names. When set, the source
            band list must equal it; when None, the manifest records the
            source bands.
        gsd_reference_m: Reference metres used by ``to_model_gsd``.
        weight_table_id: Class-weight table identifier. Empty when unused.
    """

    split: SplitRecipe = SplitRecipe()
    augment: AugmentRecipe = AugmentRecipe()
    tasks: tuple[str, ...] = TASK_NAMES
    input_bands: tuple[str, ...] | None = None
    gsd_reference_m: float = GSD_REFERENCE_M
    weight_table_id: str = ""

    @model_validator(mode="after")
    def _bounds(self) -> Self:
        """Reject an empty task list, an unknown task, a duplicate, or a bad reference."""
        if len(self.tasks) < 1:
            raise ValueError("tasks must be non-empty")
        unknown = [task for task in self.tasks if task not in TASK_NAMES]
        if unknown:
            raise ValueError(f"unknown tasks {unknown}")
        if len(set(self.tasks)) != len(self.tasks):
            raise ValueError("tasks must be unique")
        if self.input_bands is not None and len(self.input_bands) < 1:
            raise ValueError("input_bands must be non-empty")
        if not _finite_positive(self.gsd_reference_m):
            raise ValueError(f"gsd_reference_m must be finite and > 0; got {self.gsd_reference_m}")
        return self


def load_build_spec(path: str | Path) -> BuildSpec:
    """Load a BuildSpec from TOML.

    Args:
        path: TOML file. Recognized keys are ``split``, ``augment``, ``tasks``,
            ``input_bands``, ``gsd_reference_m``, and ``weight_table_id``.

    Returns:
        BuildSpec: Parsed specification. Missing keys keep the dataclass defaults.

    Raises:
        OSError / tomllib.TOMLDecodeError: On a missing or malformed file.
        ValueError: If the root is not a table, a key is unknown, or a field
            has the wrong shape.
    """
    dest = Path(path)
    raw = tomllib.loads(dest.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{dest.name} must be a TOML table")
    known = {
        "split",
        "augment",
        "tasks",
        "input_bands",
        "gsd_reference_m",
        "weight_table_id",
    }
    extra = sorted(set(raw) - known)
    if extra:
        raise ValueError(f"unknown build spec keys {extra}")
    split = _split_from_table(raw.get("split", {}))
    augment = _augment_from_table(raw.get("augment", {}))
    tasks = _string_tuple(raw.get("tasks", TASK_NAMES), "tasks")
    bands = _string_tuple(raw["input_bands"], "input_bands") if "input_bands" in raw else None
    reference = raw.get("gsd_reference_m", GSD_REFERENCE_M)
    weight = raw.get("weight_table_id", "")
    if isinstance(reference, bool) or not isinstance(reference, int | float):
        raise ValueError("gsd_reference_m must be a number")
    if not isinstance(weight, str):
        raise ValueError("weight_table_id must be a string")
    return BuildSpec(
        split=split,
        augment=augment,
        tasks=tasks,
        input_bands=bands,
        gsd_reference_m=float(reference),
        weight_table_id=weight,
    )


def _finite_positive(value: float) -> bool:
    """Return True when ``value`` is finite and greater than 0.

    Args:
        value: Candidate reference GSD.

    Returns:
        bool: True for a usable reference.
    """
    return value > 0.0 and value == value and value not in (float("inf"), float("-inf"))


def _split_from_table(value: object) -> SplitRecipe:
    """Build a SplitRecipe from a TOML table.

    Args:
        value: Decoded ``split`` table, or ``{}`` when the key is absent.

    Returns:
        SplitRecipe: Parsed recipe.

    Raises:
        ValueError: If the value is not a table of numbers.
    """
    if not isinstance(value, dict):
        raise ValueError("split must be a table")
    if set(value) - {"seed", "train_fraction", "val_fraction", "test_fraction"}:
        raise ValueError("unknown split recipe keys")
    seed = value.get("seed", 0)
    train = value.get("train_fraction", 0.70)
    val = value.get("val_fraction", 0.15)
    test = value.get("test_fraction", 0.15)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("split.seed must be an integer")
    for name, part in (
        ("train_fraction", train),
        ("val_fraction", val),
        ("test_fraction", test),
    ):
        if isinstance(part, bool) or not isinstance(part, int | float):
            raise ValueError(f"split.{name} must be a number")
    return SplitRecipe(
        seed=seed,
        train_fraction=float(train),
        val_fraction=float(val),
        test_fraction=float(test),
    )


def _augment_from_table(value: object) -> AugmentRecipe:
    """Build an AugmentRecipe from a TOML table.

    Args:
        value: Decoded ``augment`` table.

    Returns:
        AugmentRecipe: Parsed element list.

    Raises:
        ValueError: If the value is not a table or ``elements`` is not a list
            of strings.
    """
    if not isinstance(value, dict):
        raise ValueError("augment must be a table")
    if set(value) - {"elements"}:
        raise ValueError("unknown augmentation recipe keys")
    elements = _string_tuple(value.get("elements", AugmentRecipe().elements), "augment.elements")
    return AugmentRecipe(elements=elements)


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    """Return a tuple of strings.

    Args:
        value: Candidate list.
        name: Field name used in the error.

    Returns:
        tuple[str, ...]: Copied strings.

    Raises:
        ValueError: If ``value`` is not a list of strings.
    """
    if not isinstance(value, list | tuple):
        raise ValueError(f"{name} must be a list of strings")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{name} must be a list of strings")
        items.append(item)
    return tuple(items)
