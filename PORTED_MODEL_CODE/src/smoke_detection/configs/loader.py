"""YAML → typed config loader with simple dotted CLI overrides.

Precedence: YAML file < env vars (via pydantic-settings) < ``overrides`` list.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from smoke_detection.configs.base import BaseConfig
from smoke_detection.configs.classification import ClassificationConfig
from smoke_detection.configs.segmentation import SegmentationConfig

_SCHEMAS: dict[str, type[BaseConfig]] = {
    "classification": ClassificationConfig,
    "segmentation": SegmentationConfig,
}


def load_config(path: str | Path, overrides: list[str] | None = None) -> BaseConfig:
    """Load and validate a YAML config; apply ``key=value`` dotted overrides."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if overrides:
        for item in overrides:
            _apply_dotted_override(raw, item)
    task = raw.get("task")
    if task not in _SCHEMAS:
        raise ValueError(
            f"Config {path} has invalid or missing 'task': {task!r}. "
            f"Expected one of {sorted(_SCHEMAS)}."
        )
    return _SCHEMAS[task].model_validate(raw)


def _apply_dotted_override(raw: dict[str, Any], override: str) -> None:
    """Apply one ``dotted.path=value`` mutation in place on ``raw``."""
    if "=" not in override:
        raise ValueError(f"Override must be in 'key=value' form, got: {override!r}")
    key, value = override.split("=", 1)
    keys = key.split(".")
    node = raw
    for part in keys[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ValueError(f"Cannot descend into non-mapping at key: {part}")
    node[keys[-1]] = _coerce_scalar(value)


def _coerce_scalar(value: str) -> Any:
    """Coerce CLI-style override strings to int/float/bool/null when unambiguous."""
    lower = value.lower()
    if lower in ("true", "false"):
        return lower == "true"
    if lower in ("null", "none"):
        return None
    try:
        if "." in value or "e" in lower:
            return float(value)
        return int(value)
    except ValueError:
        return value
