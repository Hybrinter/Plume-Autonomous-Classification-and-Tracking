"""Plain-torch train loop for the classifier and the segmentor.

Importing this module does not import torch. Torch and torchvision load inside
`train` after the `pact-tools[train]` extra is installed.

The loop is SGD + BCEWithLogitsLoss over a frozen TrainConfig (dataclass or
TOML). Data is synthetic by default; `data_dir` selects the on-disk numpy
adapter.

Contains:
  - TrainConfig: frozen hyperparameters.
  - load_train_config: dataclass defaults overlaid with an optional TOML file.
  - overlay_train_config: CLI field overlays.
  - train: run the loop and write a checkpoint.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np

from tools.model.data import SampleBatch, load_disk_batch, make_synthetic_batch

if TYPE_CHECKING:
    from torch import nn

_TRAIN_KINDS = frozenset({"classifier", "segmentor"})


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Frozen train hyperparameters.

    Defaults match the flight inference contract (4 bands, 256 px) and a short
    SGD schedule. Spatial size is not frozen in the network; it comes from
    these fields.
    """

    kind: str = "segmentor"
    input_height_px: int = 256
    input_width_px: int = 256
    in_channels: int = 4
    epochs: int = 1
    batch_size: int = 2
    learning_rate: float = 0.01
    momentum: float = 0.9
    seed: int = 0
    synthetic_samples: int = 4
    data_dir: str = ""
    checkpoint_path: str = "artifacts/checkpoint.pt"
    bit_depth: int = 12


def load_train_config(path: str | None = None) -> TrainConfig:
    """Return TrainConfig defaults, overlaid with a TOML file when `path` is set.

    Args:
        path: Optional TOML file. Known keys match TrainConfig field names.

    Returns:
        TrainConfig: Frozen config.

    Raises:
        OSError / tomllib.TOMLDecodeError / KeyError: on a missing or malformed
        file (tools-side engineering check).
        ValueError: If `kind` is not classifier or segmentor.
    """
    cfg = TrainConfig()
    if path is None:
        return cfg
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    valid = {item.name for item in fields(TrainConfig)}
    kind = str(data["kind"]) if "kind" in data else cfg.kind
    if kind not in _TRAIN_KINDS:
        raise ValueError(f"unknown train kind {kind!r}")
    unknown = [key for key in data if key not in valid]
    if unknown:
        raise ValueError(f"unknown train config keys {unknown!r}")
    return TrainConfig(
        kind=kind,
        input_height_px=int(data.get("input_height_px", cfg.input_height_px)),
        input_width_px=int(data.get("input_width_px", cfg.input_width_px)),
        in_channels=int(data.get("in_channels", cfg.in_channels)),
        epochs=int(data.get("epochs", cfg.epochs)),
        batch_size=int(data.get("batch_size", cfg.batch_size)),
        learning_rate=float(data.get("learning_rate", cfg.learning_rate)),
        momentum=float(data.get("momentum", cfg.momentum)),
        seed=int(data.get("seed", cfg.seed)),
        synthetic_samples=int(data.get("synthetic_samples", cfg.synthetic_samples)),
        data_dir=str(data.get("data_dir", cfg.data_dir)),
        checkpoint_path=str(data.get("checkpoint_path", cfg.checkpoint_path)),
        bit_depth=int(data.get("bit_depth", cfg.bit_depth)),
    )


def overlay_train_config(
    cfg: TrainConfig,
    kind: str | None = None,
    data_dir: str | None = None,
    checkpoint_path: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    input_height_px: int | None = None,
    input_width_px: int | None = None,
    seed: int | None = None,
) -> TrainConfig:
    """Return a copy of `cfg` with any non-None CLI overlays applied.

    Args:
        cfg: Base config from defaults or TOML.
        kind: Optional kind overlay.
        data_dir: Optional disk adapter path.
        checkpoint_path: Optional checkpoint destination.
        epochs: Optional epoch count.
        batch_size: Optional batch size.
        input_height_px: Optional height.
        input_width_px: Optional width.
        seed: Optional RNG seed.

    Returns:
        TrainConfig: Frozen overlay.
    """
    return TrainConfig(
        kind=kind if kind is not None else cfg.kind,
        input_height_px=input_height_px if input_height_px is not None else cfg.input_height_px,
        input_width_px=input_width_px if input_width_px is not None else cfg.input_width_px,
        in_channels=cfg.in_channels,
        epochs=epochs if epochs is not None else cfg.epochs,
        batch_size=batch_size if batch_size is not None else cfg.batch_size,
        learning_rate=cfg.learning_rate,
        momentum=cfg.momentum,
        seed=seed if seed is not None else cfg.seed,
        synthetic_samples=cfg.synthetic_samples,
        data_dir=data_dir if data_dir is not None else cfg.data_dir,
        checkpoint_path=checkpoint_path if checkpoint_path is not None else cfg.checkpoint_path,
        bit_depth=cfg.bit_depth,
    )


def _import_torch() -> ModuleType:
    """Import torch or raise a tools-extra error."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "torch is required for tools.model.train; install pact-tools[train]"
        ) from exc
    loaded: object = torch
    if not isinstance(loaded, ModuleType):
        raise TypeError("torch import did not return a module")
    return loaded


def _build_model(kind: str, in_channels: int) -> nn.Module:
    """Construct the untrained network for `kind` (lazy arch import)."""
    if kind == "classifier":
        from tools.model.arch.classifier import build_classifier

        return build_classifier(in_channels=in_channels)
    if kind == "segmentor":
        from tools.model.arch.unet import build_segmentor

        return build_segmentor(in_channels=in_channels, out_channels=1)
    raise ValueError(f"unknown train kind {kind!r}")


def _load_samples(config: TrainConfig) -> SampleBatch:
    """Load synthetic or on-disk samples for one train run."""
    if config.data_dir:
        return load_disk_batch(config.data_dir, config.kind, bit_depth=config.bit_depth)
    return make_synthetic_batch(
        kind=config.kind,
        batch_size=max(config.synthetic_samples, config.batch_size),
        channels=config.in_channels,
        height=config.input_height_px,
        width=config.input_width_px,
        seed=config.seed,
    )


def train(config: TrainConfig | None = None) -> Path:
    """Run SGD + BCEWithLogitsLoss and write a checkpoint.

    Args:
        config: Train hyperparameters. None uses TrainConfig defaults.

    Returns:
        Path: Filesystem path of the written checkpoint.

    Raises:
        ImportError: If torch is not installed.
        ValueError: If `config.kind` is unknown.

    Notes:
        The checkpoint dict holds `kind`, `state_dict`, `in_channels`,
        `input_height_px`, and `input_width_px` for export.
    """
    cfg = config if config is not None else TrainConfig()
    if cfg.kind not in _TRAIN_KINDS:
        raise ValueError(f"unknown train kind {cfg.kind!r}")
    torch = _import_torch()
    torch.manual_seed(cfg.seed)
    model = _build_model(cfg.kind, cfg.in_channels)
    model.train()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.learning_rate,
        momentum=cfg.momentum,
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()
    samples = _load_samples(cfg)
    images = torch.from_numpy(_as_contiguous(samples.images))
    targets = torch.from_numpy(_as_contiguous(samples.targets))
    n = int(images.shape[0])
    batch = max(int(cfg.batch_size), 1)
    for _epoch in range(cfg.epochs):
        start = 0
        while start < n:
            end = min(start + batch, n)
            batch_x = images[start:end]
            batch_y = targets[start:end]
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            start = end
    out = Path(cfg.checkpoint_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": cfg.kind,
            "state_dict": model.state_dict(),
            "in_channels": cfg.in_channels,
            "input_height_px": cfg.input_height_px,
            "input_width_px": cfg.input_width_px,
        },
        out,
    )
    return out


def _as_contiguous(array: np.ndarray) -> np.ndarray:
    """Return a C-contiguous float32 numpy view for torch.from_numpy.

    Args:
        array: Array-like training tensor.

    Returns:
        np.ndarray[float32]: Contiguous copy when needed.
    """
    return np.ascontiguousarray(array, dtype=np.float32)
