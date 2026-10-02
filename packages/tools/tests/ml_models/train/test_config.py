"""Tests for the authored TrainConfig and its TOML helpers."""

from pathlib import Path

import pytest
from tools.ml_models.train.config import (
    TrainConfig,
    apply_train_mapping,
    config_digest,
    load_train_config,
    write_train_config_toml,
)


def test_defaults_validate() -> None:
    """The default configuration is well-formed."""
    cfg = TrainConfig()
    assert cfg.kind == "segmentor"
    assert cfg.datasets == ()


def test_unknown_key_rejected(tmp_path: Path) -> None:
    """Strict parsing refuses keys outside TrainConfig."""
    path = tmp_path / "bad.toml"
    path.write_text("epochs = 1\nnonsense = 2\n")
    with pytest.raises(ValueError):
        load_train_config(str(path))


def test_invalid_field_values_rejected() -> None:
    """Bounds on counts, rates, and metrics are enforced."""
    with pytest.raises(ValueError):
        TrainConfig(epochs=0)
    with pytest.raises(ValueError):
        TrainConfig(batch_size=0)
    with pytest.raises(ValueError):
        TrainConfig(learning_rate=-1.0)
    with pytest.raises(ValueError):
        TrainConfig(momentum=1.5)
    with pytest.raises(ValueError):
        TrainConfig(kind="classifier", val_metric="mean_iou")
    with pytest.raises(ValueError):
        TrainConfig(kind="segmentor", val_metric="f1")


def test_dataset_weights_must_match_datasets() -> None:
    """One positive weight per dataset, or none at all."""
    TrainConfig(datasets=("a", "b"), dataset_weights=(1.0, 2.0))
    with pytest.raises(ValueError):
        TrainConfig(datasets=("a", "b"), dataset_weights=(1.0,))
    with pytest.raises(ValueError):
        TrainConfig(datasets=("a",), dataset_weights=(0.0,))


def test_mapping_overlay(tmp_path: Path) -> None:
    """``apply_train_mapping`` replaces fields and revalidates."""
    cfg = TrainConfig()
    merged = apply_train_mapping(cfg, {"kind": "classifier", "epochs": 3})
    assert merged.kind == "classifier"
    assert merged.epochs == 3
    with pytest.raises(ValueError):
        apply_train_mapping(cfg, {"epochs": -2})


def test_config_digest_excludes_output_controls() -> None:
    """Run-path fields do not change the digest; training fields do."""
    base = TrainConfig(run_dir="a", run_id="one")
    other = TrainConfig(run_dir="b", run_id="two", checkpoint_path="c.pt", overwrite=True)
    assert config_digest(base) == config_digest(other)
    assert config_digest(base) != config_digest(TrainConfig(epochs=2))


def test_toml_round_trip(tmp_path: Path) -> None:
    """A written config parses back to the same dataclass."""
    cfg = TrainConfig(kind="classifier", epochs=2, datasets=("d1",))
    path = tmp_path / "cfg.toml"
    write_train_config_toml(path, cfg)
    loaded = load_train_config(str(path))
    assert loaded.kind == "classifier"
    assert loaded.epochs == 2
    assert loaded.datasets == ("d1",)
