"""Tests for the strict analysis, evaluation, and plot config records."""

import dataclasses
from pathlib import Path

import pytest
from flight.libs.types import Err, Ok
from tools.ml_models.analysis.config import (
    CaptureConfig,
    DatasetAnalysisConfig,
    EvaluationConfig,
    GeneralizationConfig,
    ModelAnalysisConfig,
    PlotConfig,
    ScoreConfig,
    config_digest,
    load_dataset_analysis_config,
    load_model_analysis_config,
    load_plot_config,
    render_digest,
    write_config,
)


def test_dataset_analysis_config_fields() -> None:
    """DatasetAnalysisConfig names the dataset and output directory."""
    cfg = DatasetAnalysisConfig(dataset="ds", out="out")
    assert cfg.dataset == "ds"
    assert cfg.out == "out"
    assert isinstance(cfg.plot, PlotConfig)
    assert isinstance(cfg.capture, CaptureConfig)
    assert isinstance(cfg.generalization, GeneralizationConfig)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.out = "other"  # type: ignore[misc]


def test_analysis_configs_reject_blank_paths() -> None:
    """Dataset, run, and output paths must be nonblank."""
    with pytest.raises(ValueError):
        DatasetAnalysisConfig(dataset="", out="out")
    with pytest.raises(ValueError):
        DatasetAnalysisConfig(dataset="ds", out="   ")
    with pytest.raises(ValueError):
        ModelAnalysisConfig(run="", out="out")
    with pytest.raises(ValueError):
        ModelAnalysisConfig(run="run", out="out", checkpoint="")
    with pytest.raises(ValueError):
        ModelAnalysisConfig(run="run", out="out", dataset=" ")
    ModelAnalysisConfig(run="run", out="out", dataset="ds")


def test_model_analysis_config_defaults() -> None:
    """ModelAnalysisConfig defaults to the best checkpoint without test."""
    cfg = ModelAnalysisConfig(run="run", out="out")
    assert cfg.checkpoint == "best"
    assert cfg.final_test is False
    assert cfg.dataset is None
    assert cfg.batch_size == 2
    assert cfg.device == "cpu"
    assert isinstance(cfg.score, ScoreConfig)
    assert isinstance(cfg.capture, CaptureConfig)
    assert isinstance(cfg.generalization, GeneralizationConfig)
    assert isinstance(cfg.plot, PlotConfig)


def test_model_analysis_config_strict_types() -> None:
    """Booleans and numbers do not coerce from strings or ints."""
    for bad in ("false", "true", "0", "1", 0, 1, 0.0):
        with pytest.raises(ValueError):
            ModelAnalysisConfig(run="run", out="out", final_test=bad)  # type: ignore[arg-type]
    ModelAnalysisConfig(run="run", out="out", final_test=True)


def test_evaluation_config_defaults() -> None:
    """EvaluationConfig requires kind and split; batch size and device default."""
    cfg = EvaluationConfig(kind="segmentor", split="val")
    assert cfg.batch_size == 2
    assert cfg.device == "cpu"
    assert isinstance(cfg.score, ScoreConfig)
    assert isinstance(cfg.capture, CaptureConfig)
    with pytest.raises(ValueError):
        EvaluationConfig(kind="segmentor", split="val", batch_size=0)
    with pytest.raises(ValueError):
        EvaluationConfig(kind="segmentor", split="val", batch_size=True)
    with pytest.raises(ValueError):
        EvaluationConfig(kind="segmentor", split="val", device="")


def test_plot_config_defaults() -> None:
    """PlotConfig defaults to PNG plus SVG at 300 dpi."""
    cfg = PlotConfig()
    assert cfg.formats == ("png", "svg")
    assert cfg.dpi == 300
    assert cfg.width_inches == 7.0
    assert cfg.height_inches == 4.5
    assert cfg.font_size == 11.0


def test_plot_config_validation() -> None:
    """Plot formats are literal, unique, and nonempty; sizes are positive."""
    assert PlotConfig(formats=("pdf",)).formats == ("pdf",)
    with pytest.raises(ValueError):
        PlotConfig(formats=())
    with pytest.raises(ValueError):
        PlotConfig(formats=("png", "png"))
    with pytest.raises(ValueError):
        PlotConfig(formats=("gif",))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        PlotConfig(dpi=71)
    with pytest.raises(ValueError):
        PlotConfig(dpi=True)
    with pytest.raises(ValueError):
        PlotConfig(width_inches=0.0)
    with pytest.raises(ValueError):
        PlotConfig(height_inches=float("nan"))
    with pytest.raises(ValueError):
        PlotConfig(height_inches="4.5")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        PlotConfig(font_size=-1.0)
    with pytest.raises(ValueError):
        PlotConfig(dpi="300")  # type: ignore[arg-type]


def test_score_config_defaults_and_validation() -> None:
    """ScoreConfig carries probability, matching, and histogram settings."""
    cfg = ScoreConfig()
    assert cfg.classifier_probability_threshold == 0.5
    assert cfg.blob_probability_threshold == 0.55
    assert cfg.min_blob_area_px == 15
    assert cfg.match_iou_min == 0.5
    assert cfg.boundary_tolerance_px == 1.0
    assert cfg.boundary_tolerance_m is None
    assert cfg.n_calibration_bins == 10
    assert cfg.pixel_histogram_bins == 4096
    assert cfg.f_beta == 1.0
    assert len(cfg.probability_thresholds) == 101
    assert cfg.probability_thresholds[0] == 0.0
    assert cfg.probability_thresholds[-1] == 1.0
    for kwargs in (
        {"classifier_probability_threshold": -0.1},
        {"mask_probability_threshold": 1.5},
        {"blob_probability_threshold": float("nan")},
        {"classifier_probability_threshold": True},
        {"match_iou_min": "0.5"},
        {"boundary_tolerance_m": "0.3"},
        {"min_blob_area_px": 0},
        {"min_blob_area_px": True},
        {"match_iou_min": 1.1},
        {"boundary_tolerance_px": 0.0},
        {"boundary_tolerance_m": -0.5},
        {"boundary_tolerance_m": float("inf")},
        {"n_calibration_bins": 1},
        {"pixel_histogram_bins": 1},
        {"f_beta": 0.0},
        {"probability_thresholds": ()},
        {"probability_thresholds": (0.5, 0.25)},
        {"probability_thresholds": (0.25, 0.25)},
        {"probability_thresholds": (0.25, 1.5)},
    ):
        with pytest.raises(ValueError):
            ScoreConfig(**kwargs)
    ScoreConfig(boundary_tolerance_m=0.3, probability_thresholds=(0.0, 0.5, 1.0))


def test_capture_config_validation() -> None:
    """Capture bounds are exact nonnegative ints with bytes positive."""
    cfg = CaptureConfig()
    assert cfg.retention == "COMPACT"
    assert cfg.max_preview_images == 48
    assert cfg.max_capture_bytes == 268435456
    assert cfg.examples_per_family == 12
    assert cfg.seed == 0
    with pytest.raises(ValueError):
        CaptureConfig(retention="DENSE")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        CaptureConfig(max_preview_images=-1)
    with pytest.raises(ValueError):
        CaptureConfig(max_capture_bytes=0)
    with pytest.raises(ValueError):
        CaptureConfig(examples_per_family=-1)
    with pytest.raises(ValueError):
        CaptureConfig(examples_per_family=49)
    with pytest.raises(ValueError):
        CaptureConfig(seed=-1)
    with pytest.raises(ValueError):
        CaptureConfig(seed=True)
    CaptureConfig(max_preview_images=4, examples_per_family=4)


def test_generalization_config_validation() -> None:
    """Bin edges are finite nonnegative increasing sequences."""
    cfg = GeneralizationConfig()
    assert cfg.size_edges_px == (0.0, 15.0, 64.0, 256.0, 1024.0, 4096.0)
    assert cfg.size_edges_m2 == (0.0, 1000.0, 10000.0, 100000.0, 1000000.0)
    assert cfg.gsd_edges_m == ()
    assert cfg.bootstrap_replicates == 1000
    assert cfg.confidence == 0.95
    for kwargs in (
        {"size_edges_px": (1.0,)},
        {"size_edges_px": (0.0, -1.0)},
        {"size_edges_px": (2.0, 1.0)},
        {"size_edges_px": (0.0, 0.0)},
        {"size_edges_px": (0.0, float("inf"))},
        {"size_edges_m2": ()},
        {"gsd_edges_m": (0.5,)},
        {"gsd_edges_m": (0.5, -0.1)},
        {"bootstrap_replicates": 0},
        {"confidence": 1.0},
        {"confidence": 0.0},
        {"confidence": "0.95"},
        {"confidence": True},
        {"seed": -1},
    ):
        with pytest.raises(ValueError):
            GeneralizationConfig(**kwargs)
    GeneralizationConfig(gsd_edges_m=(0.0, 0.5, 1.0))


def test_nested_toml_roundtrip(tmp_path: Path) -> None:
    """write_config and load_* round-trip nested sections exactly."""
    cfg = ModelAnalysisConfig(
        run="runs/r1",
        out="out",
        checkpoint="last",
        final_test=True,
        dataset="ds",
        batch_size=4,
        device="cuda:0",
        score=ScoreConfig(f_beta=2.0, probability_thresholds=(0.0, 0.5, 1.0)),
        capture=CaptureConfig(retention="FULL", seed=7),
        generalization=GeneralizationConfig(gsd_edges_m=(0.0, 0.5)),
        plot=PlotConfig(formats=("svg",), dpi=150),
    )
    path = tmp_path / "model.toml"
    result = write_config(path, cfg)
    assert isinstance(result, Ok)
    loaded = load_model_analysis_config(path)
    assert isinstance(loaded, Ok)
    assert loaded.value == cfg


def test_dataset_analysis_toml_roundtrip(tmp_path: Path) -> None:
    """Dataset-analysis configs round trip through nested TOML."""
    cfg = DatasetAnalysisConfig(
        dataset="data/d1",
        out="analysis/d1",
        capture=CaptureConfig(max_preview_images=8, examples_per_family=8),
    )
    path = tmp_path / "dataset.toml"
    assert isinstance(write_config(path, cfg), Ok)
    loaded = load_dataset_analysis_config(path)
    assert isinstance(loaded, Ok)
    assert loaded.value == cfg


def test_plot_config_toml_roundtrip_and_overwrite(tmp_path: Path) -> None:
    """PlotConfig round trips alone and write_config refuses overwrites."""
    path = tmp_path / "plot.toml"
    cfg = PlotConfig(formats=("pdf", "png"), dpi=200)
    assert isinstance(write_config(path, cfg), Ok)
    loaded = load_plot_config(path)
    assert isinstance(loaded, Ok)
    assert loaded.value == cfg
    again = write_config(path, cfg)
    assert isinstance(again, Err)


def test_load_config_rejects_unknown_and_missing(tmp_path: Path) -> None:
    """Loaders reject missing files, extra fields, and bad literals."""
    missing = load_plot_config(tmp_path / "none.toml")
    assert isinstance(missing, Err)
    extra = tmp_path / "extra.toml"
    extra.write_text('dpi = 300\nbogus = "x"\n', encoding="utf-8")
    assert isinstance(load_plot_config(extra), Err)
    bad = tmp_path / "bad.toml"
    bad.write_text('formats = ["gif"]\n', encoding="utf-8")
    assert isinstance(load_plot_config(bad), Err)


def test_toml_unicode_and_escape_roundtrip(tmp_path: Path) -> None:
    """Non-ASCII, non-BMP, backslash, and quote paths survive TOML codecs."""
    for dataset in ("données/éè", "dataset-\U0001f30d", 'a\\b"c', "plain"):
        cfg = DatasetAnalysisConfig(dataset=dataset, out="out")
        path = tmp_path / f"cfg-{len(dataset)}-{ord(dataset[0])}.toml"
        assert isinstance(write_config(path, cfg), Ok)
        loaded = load_dataset_analysis_config(path)
        assert isinstance(loaded, Ok)
        assert loaded.value == cfg


def test_config_digest_excludes_render_controls() -> None:
    """Scientific identity ignores out/plot; render identity includes them."""
    base = ModelAnalysisConfig(run="runs/r1", out="out-a")
    restyled = ModelAnalysisConfig(
        run="runs/r1",
        out="out-b",
        plot=PlotConfig(formats=("pdf",), dpi=72),
    )
    assert config_digest(base) == config_digest(restyled)
    other = ModelAnalysisConfig(run="runs/r2", out="out-a")
    assert config_digest(base) != config_digest(other)
    assert len(config_digest(base)) == 64
    render_a = render_digest("0" * 64, base.plot)
    render_b = render_digest("0" * 64, restyled.plot)
    assert render_a != render_b
    measurement_b = render_digest("1" * 64, base.plot)
    assert measurement_b != render_a
