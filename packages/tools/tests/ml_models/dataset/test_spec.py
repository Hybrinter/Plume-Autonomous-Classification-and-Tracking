"""Tests for the BuildSpec TOML reader."""

from pathlib import Path

import pytest
from tools.ml_models.dataset.spec import BuildSpec, load_build_spec


def test_missing_keys_keep_defaults(tmp_path: Path) -> None:
    """An empty spec file yields the dataclass defaults."""
    path = tmp_path / "spec.toml"
    path.write_text("", encoding="utf-8")
    assert load_build_spec(path) == BuildSpec()


def test_unknown_root_key_is_rejected(tmp_path: Path) -> None:
    """A stray top-level key fails the schema."""
    path = tmp_path / "spec.toml"
    path.write_text("mystery = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown build spec keys"):
        load_build_spec(path)


def test_unknown_nested_split_key_is_rejected(tmp_path: Path) -> None:
    """A stray key inside [split] fails the schema."""
    path = tmp_path / "spec.toml"
    path.write_text("[split]\nseed = 0\nstride = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown split recipe keys"):
        load_build_spec(path)


def test_unknown_nested_augment_key_is_rejected(tmp_path: Path) -> None:
    """A stray key inside [augment] fails the schema."""
    path = tmp_path / "spec.toml"
    path.write_text('[augment]\nflips = ["id"]\n', encoding="utf-8")
    with pytest.raises(ValueError, match="unknown augmentation recipe keys"):
        load_build_spec(path)


def test_tasks_must_be_known(tmp_path: Path) -> None:
    """A task outside the known names fails the schema."""
    path = tmp_path / "spec.toml"
    path.write_text('tasks = ["classifier", "detector"]\n', encoding="utf-8")
    with pytest.raises(ValueError, match="unknown tasks"):
        load_build_spec(path)
