"""Frozen configuration records for the analysis boundaries.

These dataclasses carry the typed inputs of the dataset-analysis,
model-analysis, evaluation, and render entry points. Strict validation and
resolved-config serialization land with the evidence contract phase; only the
minimal typed scaffold records are declared here.

Contains:
  - DatasetAnalysisConfig: dataset analysis inputs.
  - ModelAnalysisConfig: run analysis inputs.
  - EvaluationConfig: split evaluation inputs.
  - PlotConfig: render settings.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True, slots=True)
class DatasetAnalysisConfig:
    """Inputs for one dataset-analysis execution.

    Attributes:
        dataset: Finished dataset directory.
        out: Destination analysis directory.
    """

    dataset: str
    out: str


@dataclass(frozen=True, slots=True)
class ModelAnalysisConfig:
    """Inputs for one model/training-analysis execution.

    Attributes:
        run: Run directory.
        out: Destination analysis directory.
        checkpoint: Checkpoint selector, ``best`` by default.
        final_test: Include the final-test evaluation.
    """

    run: str
    out: str
    checkpoint: str = field(default="best")
    final_test: bool = field(default=False)


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Inputs for one exhaustive split evaluation.

    Attributes:
        kind: ``classifier`` or ``segmentor``.
        split: ``train``, ``val``, or ``test``.
        batch_size: Rows per batch.
        device: Torch device.
    """

    kind: Literal["classifier", "segmentor"]
    split: Literal["train", "val", "test"]
    batch_size: int = field(default=2)
    device: str = field(default="cpu")


@dataclass(frozen=True, slots=True)
class PlotConfig:
    """Figure output settings for one render execution.

    Attributes:
        formats: Output formats, PNG plus SVG by default.
        dpi: Raster resolution.
    """

    formats: tuple[str, ...] = field(default=("png", "svg"))
    dpi: int = field(default=300)
