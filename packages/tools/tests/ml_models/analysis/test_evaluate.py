"""Tests for the unavailable split-evaluation boundary."""

from collections.abc import Callable
from pathlib import Path

from flight.libs.types import Err
from tools.ml_models.analysis.config import EvaluationConfig
from tools.ml_models.analysis.evaluate import evaluate_split
from tools.ml_models.arch.registry import build
from tools.ml_models.dataset.manifest import load_manifest


def test_evaluate_split_returns_unavailable(
    tmp_path: Path, build_synthetic_dataset: Callable[..., Path]
) -> None:
    """A real model and dataset still cannot produce scores."""
    dataset = build_synthetic_dataset(tmp_path / "ds", n=9)
    manifest = load_manifest(dataset / "dataset.json")
    model = build("classifier", "pactnet_w8_d2", 3)
    result = evaluate_split(
        model,
        dataset,
        manifest,
        EvaluationConfig(kind="classifier", split="val"),
    )
    assert isinstance(result, Err)
    assert "unavailable" in result.error
