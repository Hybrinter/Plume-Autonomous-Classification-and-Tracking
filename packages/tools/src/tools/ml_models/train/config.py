"""Frozen train hyperparameters and pack channel checks.

Contains:
  - TrainConfig: frozen hyperparameters, including an optional canvas.
  - load_train_config: dataclass defaults overlaid with an optional TOML file.
  - overlay_train_config: CLI field overlays.
  - apply_train_mapping: overlay from a string-key mapping.
  - config_digest: 8-hex identity of experiment fields.
  - resolve_train_channels: pack channel count used for the model build.
  - write_train_config_toml: write a TrainConfig table.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, fields
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, TypeAdapter
from pydantic.dataclasses import dataclass

from tools.inference.data import ProcessedPack as InferencePack
from tools.ml_models.data.canvas import CanvasConfig
from tools.ml_models.data.pack import ProcessedPack as MlPack
from tools.ml_models.train.losses import DEFAULT_FOCAL_ALPHA, DEFAULT_FOCAL_GAMMA, LossName

TrainKind = Literal["classifier", "segmentor"]
OptimizerName = Literal["sgd", "adamw"]
SchedulerName = Literal["none", "cosine"]
_SCHEMA = ConfigDict(extra="forbid")


@dataclass(frozen=True, slots=True, config=_SCHEMA)
class TrainConfig:
    """Frozen train hyperparameters.

    Defaults are 3 bands and a 256 px spatial crop (the short training crop, not
    the flight frame) and a short SGD schedule. Spatial size is not frozen in
    the network; it comes from these fields. ``canvas`` selects flight-frame
    sampling. ``None`` keeps chip batches from a processed pack or a synthetic
    pack. ``max_steps`` caps optimizer steps. ``None`` runs every epoch.
    """

    kind: TrainKind = "segmentor"
    arch: str = ""
    input_height_px: int = 256
    input_width_px: int = 256
    in_channels: int = 3
    epochs: int = 1
    batch_size: int = 2
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0
    seed: int = 0
    synthetic_samples: int = 4
    data_dir: str = ""
    checkpoint_path: str = ""
    bit_depth: int = 12
    run_dir: str = "artifacts/runs"
    run_id: str = ""
    val_metric: str = ""
    device: str = ""
    overwrite: bool = False
    optimizer: OptimizerName = "sgd"
    scheduler: SchedulerName = "none"
    shuffle: bool = False
    pos_weight: float = 0.0
    augment: bool = False
    loss: LossName = "bce"
    focal_gamma: float = DEFAULT_FOCAL_GAMMA
    focal_alpha: float = DEFAULT_FOCAL_ALPHA
    amp: bool = False
    patience: int = 0
    eval_interval: int = 1
    max_steps: int | None = None
    canvas: CanvasConfig | None = None


_TRAIN_ADAPTER = TypeAdapter(TrainConfig)


_DIGEST_SKIP = frozenset({"run_dir", "run_id", "checkpoint_path", "overwrite"})


def _digest_value(value: object) -> object:
    """Return a JSON-ready copy of one train field."""
    if isinstance(value, CanvasConfig):
        return asdict(value)
    return value


def config_digest(cfg: TrainConfig) -> str:
    """Return an 8-hex digest of the train fields that identify an experiment.

    Args:
        cfg: Frozen train hyperparameters.

    Returns:
        str: First eight hex characters of SHA-256 over the JSON of fields
        other than ``run_dir``, ``run_id``, ``checkpoint_path``, and
        ``overwrite``.
    """
    payload = {
        item.name: _digest_value(getattr(cfg, item.name))
        for item in fields(TrainConfig)
        if item.name not in _DIGEST_SKIP
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8]


def load_train_config(path: str | None = None) -> TrainConfig:
    """Return TrainConfig defaults, overlaid with a TOML file when `path` is set.

    Args:
        path: Optional TOML file. Known keys match TrainConfig field names.
            A ``[canvas]`` table maps onto ``CanvasConfig``.

    Returns:
        TrainConfig: Frozen config.

    Raises:
        OSError / tomllib.TOMLDecodeError: on a missing or malformed file
        (tools-side engineering check).
        ValidationError: If a key is unknown or a field fails the schema.
    """
    cfg = TrainConfig()
    if path is None:
        return cfg
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    payload: dict[str, object] = dict(data)
    return apply_train_mapping(cfg, payload)


def apply_train_mapping(cfg: TrainConfig, data: dict[str, object]) -> TrainConfig:
    """Return ``cfg`` overlaid with known TrainConfig keys in ``data``.

    Args:
        cfg: Base config.
        data: Mapping of field names to TOML or JSON values.

    Returns:
        TrainConfig: Frozen overlay.

    Raises:
        ValidationError: If a key is unknown or a field fails the schema.
    """
    merged: dict[str, object] = {item.name: getattr(cfg, item.name) for item in fields(TrainConfig)}
    merged.update(data)
    return _TRAIN_ADAPTER.validate_python(merged)


def overlay_train_config(
    cfg: TrainConfig,
    kind: str | None = None,
    arch: str | None = None,
    data_dir: str | None = None,
    checkpoint_path: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    input_height_px: int | None = None,
    input_width_px: int | None = None,
    seed: int | None = None,
    run_dir: str | None = None,
    run_id: str | None = None,
    in_channels: int | None = None,
    learning_rate: float | None = None,
    momentum: float | None = None,
    weight_decay: float | None = None,
    synthetic_samples: int | None = None,
    bit_depth: int | None = None,
    val_metric: str | None = None,
    device: str | None = None,
    overwrite: bool | None = None,
    optimizer: str | None = None,
    scheduler: str | None = None,
    shuffle: bool | None = None,
    pos_weight: float | None = None,
    augment: bool | None = None,
    loss: str | None = None,
    focal_gamma: float | None = None,
    focal_alpha: float | None = None,
    amp: bool | None = None,
    patience: int | None = None,
    eval_interval: int | None = None,
    max_steps: int | None = None,
) -> TrainConfig:
    """Return a copy of `cfg` with any non-None CLI overlays applied.

    Args:
        cfg: Base config from defaults or TOML.
        kind: Optional kind overlay.
        arch: Optional architecture overlay.
        data_dir: Optional disk adapter or processed-pack path.
        checkpoint_path: Optional extra copy of last.pt.
        epochs: Optional epoch count.
        batch_size: Optional batch size.
        input_height_px: Optional height.
        input_width_px: Optional width.
        seed: Optional RNG seed.
        run_dir: Optional parent directory for runs.
        run_id: Optional run directory name.
        in_channels: Optional input band count.
        learning_rate: Optional SGD learning rate.
        momentum: Optional SGD momentum.
        weight_decay: Optional weight decay.
        synthetic_samples: Optional synthetic pack size.
        bit_depth: Optional DN bit depth.
        val_metric: Optional best-checkpoint metric name.
        device: Optional torch device string.
        overwrite: Optional replace-existing-run flag.
        optimizer: Optional ``sgd`` or ``adamw``.
        scheduler: Optional ``none`` or ``cosine``.
        shuffle: Optional train-loader shuffle flag.
        pos_weight: Optional positive-class BCE weight. ``<= 0`` disables.
        augment: Optional train-split flip and rotation flag.
        loss: Optional objective name from ``tools.ml_models.train.losses``.
        focal_gamma: Optional focal focusing exponent.
        focal_alpha: Optional focal positive-class weight.
        amp: Optional CUDA mixed-precision flag.
        patience: Optional early-stop patience in scored epochs. ``<= 0``
            disables.
        eval_interval: Optional epochs between scoring passes.
        max_steps: Optional cap on optimizer steps.

    Returns:
        TrainConfig: Frozen overlay.
    """
    candidates: dict[str, object | None] = {
        "kind": kind,
        "arch": arch,
        "data_dir": data_dir,
        "checkpoint_path": checkpoint_path,
        "epochs": epochs,
        "batch_size": batch_size,
        "input_height_px": input_height_px,
        "input_width_px": input_width_px,
        "seed": seed,
        "run_dir": run_dir,
        "run_id": run_id,
        "in_channels": in_channels,
        "learning_rate": learning_rate,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "synthetic_samples": synthetic_samples,
        "bit_depth": bit_depth,
        "val_metric": val_metric,
        "device": device,
        "overwrite": overwrite,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "shuffle": shuffle,
        "pos_weight": pos_weight,
        "augment": augment,
        "loss": loss,
        "focal_gamma": focal_gamma,
        "focal_alpha": focal_alpha,
        "amp": amp,
        "patience": patience,
        "eval_interval": eval_interval,
        "max_steps": max_steps,
    }
    updates = {key: value for key, value in candidates.items() if value is not None}
    return apply_train_mapping(cfg, updates) if updates else cfg


def resolve_train_channels(cfg: TrainConfig, pack: InferencePack | MlPack) -> int:
    """Return the input channel count to use for model build and checkpoints.

    Args:
        cfg: Frozen train hyperparameters.
        pack: Processed pack. ``in_channels`` must equal ``pack.meta.in_channels``
            and the image tensor's channel axis.

    Returns:
        int: Resolved channel count. It equals both the config and the pack.

    Raises:
        ValueError: If pack metadata disagrees with image tensors, or if
            ``cfg.in_channels`` disagrees with the pack.
    """
    pack_channels = int(pack.meta.in_channels)
    image_channels = int(pack.images.shape[1])
    if pack_channels != image_channels:
        raise ValueError(
            f"pack meta in_channels={pack_channels} does not match "
            f"images shape channels={image_channels}"
        )
    if cfg.in_channels != pack_channels:
        raise ValueError(
            f"in_channels={cfg.in_channels} does not match pack channel count {pack_channels}"
        )
    return pack_channels


def _toml_scalar(value: object) -> str:
    """Return one TOML scalar for a train field."""
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def write_train_config_toml(path: Path, cfg: TrainConfig) -> None:
    """Write TrainConfig fields as a TOML document.

    Args:
        path: Destination file.
        cfg: Frozen train hyperparameters.

    Returns:
        None.

    Notes:
        ``None`` fields are omitted. ``load_train_config`` then keeps the
        dataclass default. A set ``canvas`` is a ``[canvas]`` table.
    """
    lines: list[str] = []
    for item in fields(TrainConfig):
        value: object = getattr(cfg, item.name)
        if value is None:
            continue
        if isinstance(value, CanvasConfig):
            lines.append("[canvas]")
            for sub in fields(CanvasConfig):
                sub_value: object = getattr(value, sub.name)
                if sub.name == "frame_hw":
                    height, width = value.frame_hw
                    lines.append(f"frame_hw = [{height}, {width}]")
                else:
                    lines.append(f"{sub.name} = {_toml_scalar(sub_value)}")
            continue
        lines.append(f"{item.name} = {_toml_scalar(value)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
