"""Strict configuration records and TOML codecs for the analysis boundaries.

These frozen records carry the typed inputs of the dataset-analysis,
model-analysis, evaluation, and render entry points. Nested settings serialize
as TOML sections; unknown fields are rejected. Scientific identity excludes
output paths and render settings so figures can be restyled without changing
measurements.

Contains:
  - PlotConfig, ScoreConfig, CaptureConfig, GeneralizationConfig: settings.
  - EvaluationConfig: split evaluation inputs.
  - DatasetAnalysisConfig, ModelAnalysisConfig: analysis inputs.
  - load_*_config / write_config: strict TOML codecs.
  - config_digest / render_digest: scientific and render identities.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, field
from pathlib import Path
from typing import Literal

from flight.libs.types import Err, Ok, Result
from pydantic import ConfigDict, StrictBool, StrictInt, TypeAdapter, model_validator
from pydantic.dataclasses import dataclass

from tools.ml_models.analysis.contracts import FiniteNumber, Split, Task, is_sha256

_SCHEMA = ConfigDict(extra="forbid")


def _require_nonblank(value: str, name: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must be nonblank")


@dataclass(frozen=True, slots=True, config=_SCHEMA)
class PlotConfig:
    """Figure output settings for one render execution.

    Attributes:
        formats: Output formats, unique and nonempty.
        dpi: Raster resolution.
        width_inches: Figure width.
        height_inches: Figure height.
        font_size: Base font size.
    """

    formats: tuple[Literal["png", "svg", "pdf"], ...] = ("png", "svg")
    dpi: StrictInt = 300
    width_inches: FiniteNumber = 7.0
    height_inches: FiniteNumber = 4.5
    font_size: FiniteNumber = 11.0

    @model_validator(mode="after")
    def _bounds(self) -> PlotConfig:
        if not self.formats:
            raise ValueError("formats must be nonempty")
        if len(set(self.formats)) != len(self.formats):
            raise ValueError("formats must be unique")
        if self.dpi < 72:
            raise ValueError("dpi must be >= 72")
        if self.width_inches <= 0.0 or self.height_inches <= 0.0:
            raise ValueError("figure dimensions must be positive")
        if self.font_size <= 0.0:
            raise ValueError("font_size must be positive")
        return self


@dataclass(frozen=True, slots=True, config=_SCHEMA)
class ScoreConfig:
    """Probability, matching, and histogram settings for scoring.

    Attributes:
        classifier_probability_threshold: Positive-class decision threshold.
        mask_probability_threshold: Mask binarization threshold.
        blob_probability_threshold: Component acceptance threshold.
        min_blob_area_px: Minimum component area in pixels.
        match_iou_min: Minimum IoU for a positive component match.
        boundary_tolerance_px: Boundary tolerance in pixels.
        boundary_tolerance_m: Boundary tolerance in metres, if overridden.
        n_calibration_bins: Reliability curve bins.
        pixel_histogram_bins: Pixel-value histogram bins.
        f_beta: F-beta weight.
        probability_thresholds: Unique increasing operating-point grid.
    """

    classifier_probability_threshold: FiniteNumber = 0.5
    mask_probability_threshold: FiniteNumber = 0.5
    blob_probability_threshold: FiniteNumber = 0.55
    min_blob_area_px: StrictInt = 15
    match_iou_min: FiniteNumber = 0.5
    boundary_tolerance_px: FiniteNumber = 1.0
    boundary_tolerance_m: FiniteNumber | None = None
    n_calibration_bins: StrictInt = 10
    pixel_histogram_bins: StrictInt = 4096
    f_beta: FiniteNumber = 1.0
    probability_thresholds: tuple[FiniteNumber, ...] = field(
        default_factory=lambda: tuple(i / 100 for i in range(101))
    )

    @model_validator(mode="after")
    def _bounds(self) -> ScoreConfig:
        for value, name in (
            (self.classifier_probability_threshold, "classifier_probability_threshold"),
            (self.mask_probability_threshold, "mask_probability_threshold"),
            (self.blob_probability_threshold, "blob_probability_threshold"),
            (self.match_iou_min, "match_iou_min"),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.min_blob_area_px < 1:
            raise ValueError("min_blob_area_px must be >= 1")
        if self.boundary_tolerance_px <= 0.0:
            raise ValueError("boundary_tolerance_px must be positive")
        if self.boundary_tolerance_m is not None and self.boundary_tolerance_m <= 0.0:
            raise ValueError("boundary_tolerance_m must be positive when set")
        if self.n_calibration_bins < 2 or self.pixel_histogram_bins < 2:
            raise ValueError("histogram and calibration bins must be >= 2")
        if self.f_beta <= 0.0:
            raise ValueError("f_beta must be positive")
        _check_threshold_grid(self.probability_thresholds)
        return self


def _check_threshold_grid(values: tuple[FiniteNumber, ...]) -> None:
    if not values:
        raise ValueError("probability_thresholds must be nonempty")
    if any(not 0.0 <= v <= 1.0 for v in values):
        raise ValueError("probability_thresholds must lie in [0, 1]")
    if any(b <= a for a, b in zip(values, values[1:])):
        raise ValueError("probability_thresholds must be unique and increasing")


@dataclass(frozen=True, slots=True, config=_SCHEMA)
class CaptureConfig:
    """Capture bounds for per-row evidence retention.

    Attributes:
        retention: ``COMPACT`` keeps aggregates/previews; ``FULL`` adds dense
            predictions.
        max_preview_images: Preview budget.
        max_capture_bytes: Byte budget for persisted evidence.
        examples_per_family: Per-family preview budget.
        seed: Preview selection seed.
    """

    retention: Literal["COMPACT", "FULL"] = "COMPACT"
    max_preview_images: StrictInt = 48
    max_capture_bytes: StrictInt = 268435456
    examples_per_family: StrictInt = 12
    seed: StrictInt = 0

    @model_validator(mode="after")
    def _bounds(self) -> CaptureConfig:
        if self.max_preview_images < 0 or self.examples_per_family < 0:
            raise ValueError("preview counts must be nonnegative")
        if self.max_capture_bytes < 1:
            raise ValueError("max_capture_bytes must be positive")
        if self.examples_per_family > self.max_preview_images:
            raise ValueError("examples_per_family cannot exceed max_preview_images")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        return self


@dataclass(frozen=True, slots=True, config=_SCHEMA)
class GeneralizationConfig:
    """Stratification edges and interval budget for generalization checks.

    Attributes:
        size_edges_px: Component-size bin edges in pixels.
        size_edges_m2: Component-size bin edges in square metres.
        gsd_edges_m: GSD bin edges in metres; empty means unsliced.
        bootstrap_replicates: Replicate budget for intervals.
        confidence: Nominal interval confidence.
        seed: Replicate seed.
    """

    size_edges_px: tuple[FiniteNumber, ...] = (0.0, 15.0, 64.0, 256.0, 1024.0, 4096.0)
    size_edges_m2: tuple[FiniteNumber, ...] = (0.0, 1000.0, 10000.0, 100000.0, 1000000.0)
    gsd_edges_m: tuple[FiniteNumber, ...] = ()
    bootstrap_replicates: StrictInt = 1000
    confidence: FiniteNumber = 0.95
    seed: StrictInt = 0

    @model_validator(mode="after")
    def _bounds(self) -> GeneralizationConfig:
        for edges, name in (
            (self.size_edges_px, "size_edges_px"),
            (self.size_edges_m2, "size_edges_m2"),
            (self.gsd_edges_m, "gsd_edges_m"),
        ):
            _check_edges(edges, name)
        if self.bootstrap_replicates < 1:
            raise ValueError("bootstrap_replicates must be positive")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must lie strictly inside (0, 1)")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        return self


def _check_edges(edges: tuple[FiniteNumber, ...], name: str) -> None:
    if name != "gsd_edges_m" and len(edges) < 2:
        raise ValueError(f"{name} requires at least two edges")
    if name == "gsd_edges_m" and len(edges) == 1:
        raise ValueError("gsd_edges_m must be empty or have at least two edges")
    if any(edge < 0.0 for edge in edges):
        raise ValueError(f"{name} edges must be nonnegative")
    if any(b <= a for a, b in zip(edges, edges[1:])):
        raise ValueError(f"{name} edges must be strictly increasing")


@dataclass(frozen=True, slots=True, config=_SCHEMA)
class EvaluationConfig:
    """Inputs for one exhaustive split evaluation.

    Attributes:
        kind: ``classifier`` or ``segmentor``.
        split: ``train``, ``val``, or ``test``.
        batch_size: Rows per batch.
        device: Torch device.
        score: Scoring settings.
        capture: Capture bounds.
    """

    kind: Task
    split: Split
    batch_size: StrictInt = 2
    device: str = "cpu"
    score: ScoreConfig = field(default_factory=ScoreConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)

    @model_validator(mode="after")
    def _bounds(self) -> EvaluationConfig:
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        _require_nonblank(self.device, "device")
        return self


@dataclass(frozen=True, slots=True, config=_SCHEMA)
class DatasetAnalysisConfig:
    """Inputs for one dataset-analysis execution.

    Attributes:
        dataset: Finished dataset directory.
        out: Destination analysis directory.
        plot: Figure output settings.
        capture: Capture bounds.
        generalization: Stratification and interval settings.
    """

    dataset: str
    out: str
    plot: PlotConfig = field(default_factory=PlotConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    generalization: GeneralizationConfig = field(default_factory=GeneralizationConfig)

    @model_validator(mode="after")
    def _bounds(self) -> DatasetAnalysisConfig:
        _require_nonblank(self.dataset, "dataset")
        _require_nonblank(self.out, "out")
        return self


@dataclass(frozen=True, slots=True, config=_SCHEMA)
class ModelAnalysisConfig:
    """Inputs for one model/training-analysis execution.

    Attributes:
        run: Run directory.
        out: Destination analysis directory.
        checkpoint: Checkpoint selector, ``best`` by default.
        final_test: Include the final-test evaluation.
        dataset: Optional evaluation-only dataset override.
        batch_size: Rows per batch.
        device: Torch device.
        score: Scoring settings.
        capture: Capture bounds.
        generalization: Stratification and interval settings.
        plot: Figure output settings.
    """

    run: str
    out: str
    checkpoint: str = "best"
    final_test: StrictBool = False
    dataset: str | None = None
    batch_size: StrictInt = 2
    device: str = "cpu"
    score: ScoreConfig = field(default_factory=ScoreConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    generalization: GeneralizationConfig = field(default_factory=GeneralizationConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)

    @model_validator(mode="after")
    def _bounds(self) -> ModelAnalysisConfig:
        _require_nonblank(self.run, "run")
        _require_nonblank(self.out, "out")
        _require_nonblank(self.checkpoint, "checkpoint")
        if self.dataset is not None:
            _require_nonblank(self.dataset, "dataset")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        _require_nonblank(self.device, "device")
        return self


def _load[T](path: Path, adapter: TypeAdapter[T]) -> Result[T, str]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        return Err(f"cannot read config {path}: {exc}")
    try:
        data = tomllib.loads(raw)
    except tomllib.TOMLDecodeError as exc:
        return Err(f"invalid TOML in {path}: {exc}")
    try:
        return Ok(adapter.validate_python(data))
    except ValueError as exc:
        return Err(f"invalid config {path}: {exc}")


def load_dataset_analysis_config(path: Path) -> Result[DatasetAnalysisConfig, str]:
    """Read a strict TOML dataset-analysis configuration."""
    return _load(path, TypeAdapter(DatasetAnalysisConfig))


def load_model_analysis_config(path: Path) -> Result[ModelAnalysisConfig, str]:
    """Read a strict TOML model/training-analysis configuration."""
    return _load(path, TypeAdapter(ModelAnalysisConfig))


def load_plot_config(path: Path) -> Result[PlotConfig, str]:
    """Read a strict TOML render configuration."""
    return _load(path, TypeAdapter(PlotConfig))


def _toml_scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (tuple, list)):
        return "[" + ", ".join(_toml_scalar(v) for v in value) + "]"
    raise TypeError(f"cannot serialize TOML value {value!r}")


def _toml_lines(mapping: dict[str, object], prefix: str) -> list[str]:
    lines: list[str] = []
    scalars = {k: v for k, v in mapping.items() if not isinstance(v, dict)}
    sections = {k: v for k, v in mapping.items() if isinstance(v, dict)}
    if prefix:
        lines.append(f"[{prefix}]")
    for key in sorted(scalars):
        value = scalars[key]
        if value is not None:
            lines.append(f"{key} = {_toml_scalar(value)}")
    for key in sorted(sections):
        lines.append("")
        lines.extend(_toml_lines(sections[key], f"{prefix}.{key}" if prefix else key))
    return lines


def write_config(
    path: Path,
    cfg: DatasetAnalysisConfig | ModelAnalysisConfig | PlotConfig,
) -> Result[None, str]:
    """Write ``cfg`` as nested TOML, refusing to overwrite an existing file."""
    try:
        mapping = asdict(cfg)
    except TypeError as exc:
        return Err(f"cannot serialize config: {exc}")
    text = "\n".join(_toml_lines(mapping, "")) + "\n"
    try:
        with Path(path).open("x", encoding="utf-8") as handle:
            handle.write(text)
    except FileExistsError:
        return Err(f"config {path} already exists; refusing to overwrite")
    except OSError as exc:
        return Err(f"cannot write config {path}: {exc}")
    return Ok(None)


def _scientific_values(cfg: DatasetAnalysisConfig | ModelAnalysisConfig) -> dict[str, object]:
    values = asdict(cfg)
    values.pop("out")
    values.pop("plot")
    return values


def config_digest(cfg: DatasetAnalysisConfig | ModelAnalysisConfig) -> str:
    """Hash the scientific settings, excluding output paths and render style."""
    values = _scientific_values(cfg)
    payload = json.dumps(values, sort_keys=True, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def render_digest(measurement_id: str, plot: PlotConfig) -> str:
    """Hash the render identity: frozen measurement id plus plot settings."""
    if not is_sha256(measurement_id):
        raise ValueError("measurement_id must be a lowercase SHA-256 hex string")
    payload = json.dumps(
        {"measurement_id": measurement_id, "plot": asdict(plot)},
        sort_keys=True,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
