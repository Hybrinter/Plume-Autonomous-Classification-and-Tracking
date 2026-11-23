"""Tests for the authored TrainConfig and its TOML helpers."""

from pathlib import Path
from typing import Any, cast

import pytest
from tools.ml_models.train.config import (
    TrainConfig,
    apply_train_mapping,
    config_digest,
    load_train_config,
    validation_metric,
    write_train_config_toml,
)


def test_defaults_validate() -> None:
    """The default configuration is well-formed."""
    cfg = TrainConfig()
    assert cfg.kind == "segmentor"
    assert cfg.dataset == ""


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


def test_legacy_plural_dataset_keys_rejected(tmp_path: Path) -> None:
    """The old ``datasets`` and ``dataset_weights`` keys fail with guidance."""
    cfg = TrainConfig()
    for overlay in (
        cast(dict[str, object], {"datasets": ("a",)}),
        cast(dict[str, object], {"dataset_weights": (1.0,)}),
    ):
        with pytest.raises(ValueError, match="dataset"):
            apply_train_mapping(cfg, overlay)
        with pytest.raises(ValueError, match="dataset"):
            apply_train_mapping(cfg, {"dataset": "a"} | overlay)
    path = tmp_path / "legacy.toml"
    path.write_text('datasets = ["a", "b"]\n')
    with pytest.raises(ValueError, match="dataset"):
        load_train_config(str(path))


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
    cfg = TrainConfig(kind="classifier", epochs=2, dataset="d1")
    path = tmp_path / "cfg.toml"
    write_train_config_toml(path, cfg)
    loaded = load_train_config(str(path))
    assert loaded.kind == "classifier"
    assert loaded.epochs == 2
    assert loaded.dataset == "d1"


def test_validation_metric_resolves_defaults_and_aliases() -> None:
    """Defaults and legacy aliases map to defined scoring names."""
    assert validation_metric("classifier") == "average_precision"
    assert validation_metric("classifier", "pr_auc") == "average_precision"
    assert validation_metric("classifier", "bce") == "binary_cross_entropy"
    assert validation_metric("classifier", "brier") == "brier_score"
    assert validation_metric("segmentor") == "foreground_iou_mean_positive_images"
    assert validation_metric("segmentor", "mean_iou") == (
        "foreground_iou_mean_all_annotated_images"
    )
    assert validation_metric("segmentor", "mean_dice") == (
        "foreground_dice_mean_all_annotated_images"
    )
    assert validation_metric("segmentor", "bce") == "binary_cross_entropy_mean_images"
    assert validation_metric("classifier", "roc_auc") == "roc_auc"


def test_validation_metric_bounds() -> None:
    """Alias inputs validate; undefined or cross-task metrics are rejected."""
    assert TrainConfig(kind="classifier", val_metric="bce").val_metric == "bce"
    assert TrainConfig(kind="classifier", val_metric="pr_auc").val_metric == "pr_auc"
    assert TrainConfig(kind="segmentor", val_metric="mean_iou").val_metric == "mean_iou"
    assert TrainConfig(kind="segmentor", val_metric="foreground_iou_mean_positive_images")
    with pytest.raises(ValueError):
        TrainConfig(kind="classifier", val_metric="not_a_metric")
    with pytest.raises(ValueError):
        TrainConfig(kind="segmentor", val_metric="average_precision")
    with pytest.raises(ValueError):
        TrainConfig(kind="classifier", val_metric="mean_iou")


def test_checkpoint_selection_and_diagnostics_fields() -> None:
    """New training controls default safely and validate their literals."""
    cfg = TrainConfig()
    assert cfg.selected_checkpoint == "best"
    assert cfg.gradient_diagnostics is False
    assert cfg.amp_diagnostics is False
    assert TrainConfig(selected_checkpoint="last").selected_checkpoint == "last"
    with pytest.raises(ValueError):
        TrainConfig(selected_checkpoint=cast(Any, "bestest"))
