"""Shared pydantic config schemas. Task-specific configs subclass ``BaseConfig``."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict


class TrainerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_epochs: int = 50
    accelerator: Literal["auto", "cpu", "gpu"] = "auto"
    devices: int | str = "auto"
    precision: Literal["32", "16-mixed", "bf16-mixed"] = "32"
    deterministic: bool = True
    gradient_clip_val: float | None = None
    log_every_n_steps: int = 10
    fast_dev_run: bool = False


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_root: Path = Path("data")
    output_dir: Path = Path("lightning_logs")
    experiment_name: str


class OptimConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lr: float
    momentum: float = 0.9
    weight_decay: float = 0.0
    scheduler: Literal["none", "cosine", "plateau"] = "none"


class BaseConfig(BaseSettings):
    """Top-level config. Subclasses add ``model:`` and ``data:`` blocks."""

    model_config = SettingsConfigDict(
        env_prefix="SMOKEDET_",
        env_nested_delimiter="__",
        extra="forbid",
    )

    task: Literal["classification", "segmentation"]
    seed: int = 42
    trainer: TrainerConfig
    paths: PathsConfig
    optim: OptimConfig
