"""Configuration for training GSD-conditioned models on finished datasets."""

from __future__ import annotations

import hashlib
import json
import math
import tomllib
from dataclasses import asdict
from pathlib import Path
from typing import Literal, Self

from flight.libs.types import Err
from pydantic import ConfigDict, TypeAdapter, model_validator
from pydantic.dataclasses import dataclass

from tools.ml_models.analysis.metrics.definitions import metric_definition


def validation_metric(kind: str, name: str = "") -> str:
    """Resolve defaults and explicit legacy aliases to defined scoring names."""
    if kind == "classifier":
        return {
            "": "average_precision",
            "pr_auc": "average_precision",
            "bce": "binary_cross_entropy",
            "brier": "brier_score",
        }.get(name, name)
    return {
        "": "foreground_iou_mean_positive_images",
        "mean_iou": "foreground_iou_mean_all_annotated_images",
        "mean_dice": "foreground_dice_mean_all_annotated_images",
        "bce": "binary_cross_entropy_mean_images",
    }.get(name, name)


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class TrainConfig:
    """Training paths and optimizer controls; datasets own preprocessing and augmentation."""

    kind: Literal["classifier", "segmentor"] = "segmentor"
    arch: str = ""
    dataset: str = ""
    epochs: int = 1
    batch_size: int = 2
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0
    seed: int = 0
    run_dir: str = "artifacts/runs"
    run_id: str = ""
    checkpoint_path: str = ""
    device: str = ""
    overwrite: bool = False
    optimizer: Literal["sgd", "adamw"] = "sgd"
    scheduler: Literal["none", "cosine"] = "none"
    loss: Literal["bce", "dice", "bce_dice", "focal", "focal_dice"] = "bce"
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    pos_weight: float = 0.0
    amp: bool = False
    patience: int = 0
    eval_interval: int = 1
    max_steps: int | None = None
    val_metric: str = ""
    selected_checkpoint: Literal["best", "last"] = "best"
    gradient_diagnostics: bool = False
    amp_diagnostics: bool = False

    @model_validator(mode="before")
    @classmethod
    def _single_dataset(cls, data: object) -> object:
        if isinstance(data, dict):
            legacy = [key for key in ("datasets", "dataset_weights") if key in data]
            if legacy:
                raise ValueError(
                    f"training uses exactly one dataset; set 'dataset' instead of {sorted(legacy)}"
                )
        return data

    @model_validator(mode="after")
    def _bounds(self) -> Self:
        if min(self.epochs, self.batch_size, self.eval_interval) < 1:
            raise ValueError("epochs, batch_size, and eval_interval must be positive")
        if self.patience < 0 or (self.max_steps is not None and self.max_steps < 1):
            raise ValueError("invalid patience or max_steps")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if not math.isfinite(self.momentum) or not 0 <= self.momentum < 1:
            raise ValueError("momentum must lie in [0,1)")
        if (
            not math.isfinite(self.focal_gamma)
            or self.focal_gamma < 0
            or not math.isfinite(self.focal_alpha)
            or not 0 <= self.focal_alpha <= 1
            or not math.isfinite(self.pos_weight)
            or self.pos_weight < 0
        ):
            raise ValueError("invalid focal or positive-class loss weights")
        allowed = (
            {
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "average_precision",
                "brier_score",
                "binary_cross_entropy",
            }
            if self.kind == "classifier"
            else {
                "foreground_iou_mean_positive_images",
                "foreground_dice_mean_positive_images",
                "foreground_iou_mean_all_annotated_images",
                "foreground_dice_mean_all_annotated_images",
                "foreground_iou_global",
                "foreground_dice_global",
                "binary_cross_entropy_mean_images",
                "brier_score_mean_images",
            }
        )
        metric = validation_metric(self.kind, self.val_metric)
        definition = metric_definition(metric)
        if metric not in allowed or isinstance(definition, Err):
            raise ValueError("validation metric is invalid for the model task")
        return self


def load_train_config(path: str | None = None) -> TrainConfig:
    """Read defaults or a strict flat TOML training configuration."""
    if path is None:
        return TrainConfig()
    return TypeAdapter(TrainConfig).validate_python(tomllib.loads(Path(path).read_text()))


def apply_train_mapping(cfg: TrainConfig, data: dict[str, object]) -> TrainConfig:
    """Validate a configuration overlay."""
    return TypeAdapter(TrainConfig).validate_python(asdict(cfg) | data)


def config_digest(cfg: TrainConfig) -> str:
    """Hash experiment fields, excluding output-path controls."""
    values = asdict(cfg)
    for key in ("run_dir", "run_id", "checkpoint_path", "overwrite"):
        values.pop(key)
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()[:8]


def write_train_config_toml(path: str | Path, cfg: TrainConfig) -> None:
    """Write a flat TOML configuration, omitting optional None values."""
    lines = []
    for key, value in asdict(cfg).items():
        if value is not None:
            lines.append(f"{key} = {json.dumps(value)}")
    Path(path).write_text("\n".join(lines) + "\n")
