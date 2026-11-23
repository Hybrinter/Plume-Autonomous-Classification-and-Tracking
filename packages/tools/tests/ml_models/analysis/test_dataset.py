"""Tests for dataset measurement helpers and publication."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from flight.libs.types import Err, Ok
from tools.ml_models.analysis.artifacts import verify_bundle
from tools.ml_models.analysis.config import (
    CaptureConfig,
    DatasetAnalysisConfig,
    PlotConfig,
)
from tools.ml_models.analysis.dataset import analyze_dataset, measure_mask, measure_pixels
from tools.ml_models.analysis.summaries import DatasetSummary
from tools.ml_models.dataset.manifest import compute_dataset_hash


@pytest.mark.slow
def test_analyze_dataset_publishes_verifiable_bundle(
    tmp_path: Path, build_synthetic_dataset: Callable[..., Path]
) -> None:
    """A finished dataset publishes rendered figures, visuals, and previews."""
    dataset = build_synthetic_dataset(tmp_path / "ds", n=9)
    before = compute_dataset_hash(dataset)
    out = tmp_path / "analysis-out"
    result = analyze_dataset(DatasetAnalysisConfig(dataset=str(dataset), out=str(out)))
    assert isinstance(result, Ok)
    verified = verify_bundle(result.value)
    assert isinstance(verified, Ok)
    summary = verified.value
    assert isinstance(summary, DatasetSummary)
    rendering = json.loads((out / "rendering.json").read_text(encoding="utf-8"))
    assert rendering["measurement_id"] == summary.measurement_id
    assert rendering["render_id"] != summary.measurement_id
    assert compute_dataset_hash(dataset) == before
    paths = {ref.path for ref in summary.artifacts}
    assert {"figure-data.json", "preview-manifest.json", "rendering.json"} <= paths
    assert any(path.startswith("figures/") for path in paths)
    assert any(path.startswith("visuals/") for path in paths)
    outputs = {record.name: record for record in summary.outputs}
    assert outputs["figures"].status == "AVAILABLE"
    assert outputs["timestamps"].status == "UNAVAILABLE"
    assert outputs["timestamps"].reason
    assert outputs["conditions"].status == "UNAVAILABLE"
    assert outputs["conditions"].reason
    figure_outputs = {
        name[len("figure:") :]: record
        for name, record in outputs.items()
        if name.startswith("figure:")
    }
    assert figure_outputs
    assert all(record.status != "SKIPPED" for record in figure_outputs.values())
    unavailable = {
        name: record for name, record in figure_outputs.items() if record.status == "UNAVAILABLE"
    }
    assert unavailable
    assert all(record.reason for record in unavailable.values())
    for name in unavailable:
        assert f"figures/{name}.png" in paths
        assert f"figures/{name}.svg" in paths


def test_analyze_dataset_cheap_config_publishes_indexed_bundle(
    tmp_path: Path, build_synthetic_dataset: Callable[..., Path]
) -> None:
    """A minimal plot/capture config still renders figures and indexes skipped galleries."""
    dataset = build_synthetic_dataset(tmp_path / "ds", n=3)
    out = tmp_path / "analysis-out"
    cfg = DatasetAnalysisConfig(
        dataset=str(dataset),
        out=str(out),
        plot=PlotConfig(formats=("png",), dpi=72, width_inches=7.0, height_inches=4.5),
        capture=CaptureConfig(max_preview_images=0, examples_per_family=0),
    )
    result = analyze_dataset(cfg)
    assert isinstance(result, Ok)
    verified = verify_bundle(result.value)
    assert isinstance(verified, Ok)
    summary = verified.value
    assert isinstance(summary, DatasetSummary)
    outputs = {record.name: record for record in summary.outputs}
    assert outputs["figures"].status == "AVAILABLE"
    gallery_outputs = [record for name, record in outputs.items() if name.startswith("gallery:")]
    assert gallery_outputs
    assert all(record.status != "AVAILABLE" and record.reason for record in gallery_outputs)
    assert outputs["gallery:representative"].status == "SKIPPED"
    paths = {ref.path for ref in summary.artifacts}
    assert "visuals/gallery_unavailable_representative.png" in paths
    assert not any(path.startswith("previews/") for path in paths)


def test_analyze_dataset_render_failure_leaves_no_output(
    tmp_path: Path,
    build_synthetic_dataset: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A render-stage failure aborts publication before the output is reserved."""
    import tools.ml_models.analysis.dataset_render as render_module

    dataset = build_synthetic_dataset(tmp_path / "ds", n=3)
    out = tmp_path / "analysis-out"

    def fail(measured: object, cfg: object) -> Err[str]:
        return Err("simulated render failure")

    monkeypatch.setattr(render_module, "publish_rendered_dataset", fail)
    result = analyze_dataset(DatasetAnalysisConfig(dataset=str(dataset), out=str(out)))
    assert isinstance(result, Err)
    assert "simulated render failure" in result.error
    assert not out.exists()


def test_analyze_dataset_figure_render_failure_leaves_no_output(
    tmp_path: Path,
    build_synthetic_dataset: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A figure-renderer Err inside the orchestrator aborts before reservation."""
    import tools.ml_models.analysis.dataset_render as render_module

    dataset = build_synthetic_dataset(tmp_path / "ds", n=3)
    out = tmp_path / "analysis-out"

    def fail(figures: object, cfg: object) -> Err[str]:
        return Err("simulated figure codec failure")

    monkeypatch.setattr(render_module, "render_dataset_figures", fail)
    cfg = DatasetAnalysisConfig(
        dataset=str(dataset),
        out=str(out),
        plot=PlotConfig(formats=("png",), dpi=72, width_inches=7.0, height_inches=4.5),
        capture=CaptureConfig(max_preview_images=0, examples_per_family=0),
    )
    result = analyze_dataset(cfg)
    assert isinstance(result, Err)
    assert "figure codec" in result.error
    assert not out.exists()


def test_mask_measurement_counts_unfiltered_four_connected_components() -> None:
    mask = np.zeros((1, 3, 4), dtype=np.uint8)
    mask[0, 0, 0:2] = 1
    mask[0, 1, 2] = 1
    mask[0, 2, 3] = 1
    measured = measure_mask(mask, (2.0, 3.0))
    assert isinstance(measured, Ok)
    result = measured.value
    assert result.area_px == 4
    assert result.area_fraction == pytest.approx(1 / 3)
    assert result.area_m2 == 24.0
    assert result.n_components == 3
    assert result.component_areas_px == (2, 1, 1)
    assert result.border_touching


def test_empty_explicit_mask_and_missing_mask_are_not_the_same() -> None:
    measured = measure_mask(np.zeros((1, 2, 3), dtype=np.uint8), (2.0, 3.0))
    assert isinstance(measured, Ok)
    assert measured.value.area_px == 0
    assert measured.value.area_m2 == 0
    assert measured.value.component_areas_px == ()
    assert not measured.value.border_touching
    assert isinstance(measure_mask(cast(Any, None), (2.0, 3.0)), Err)
    assert isinstance(measure_mask(np.full((1, 2, 3), 2, dtype=np.uint8), (2.0, 3.0)), Err)
    assert isinstance(measure_mask(np.zeros((1, 2, 3), dtype=np.uint8), (0.0, 3.0)), Err)


def test_pixel_moments_are_pixel_weighted_not_image_weighted() -> None:
    small = np.array([[[0.0, 1.0]], [[1.0, 0.0]]], dtype=np.float32)
    large = np.array([[[0.5, 0.5, 0.5, 0.5]], [[0.5, 0.5, 0.5, 0.5]]], dtype=np.float32)
    measured = measure_pixels((small, large), ("RED", "BLUE"), histogram_bins=2)
    assert isinstance(measured, Ok)
    result = measured.value
    assert result.n_images == 2 and result.n_pixels == 6
    assert result.histogram_bin_edges == (0.0, 0.5, 1.0)
    assert result.bands[0].mean == 0.5
    assert result.bands[0].std == pytest.approx(np.sqrt(1 / 12))
    assert (result.bands[0].minimum, result.bands[0].maximum) == (0.0, 1.0)
    assert result.bands[0].at_zero == 1 and result.bands[0].at_one == 1
    assert result.bands[0].histogram_counts == (1, 5)
    assert result.correlations[0][1] == pytest.approx(-1.0)
    assert result.correlations[0][0] == pytest.approx(1.0)


def test_constant_band_correlation_is_unavailable_not_fabricated() -> None:
    image = np.array([[[0.0, 1.0]], [[0.5, 0.5]]], dtype=np.float32)
    measured = measure_pixels((image,), ("RED", "NIR"))
    assert isinstance(measured, Ok)
    assert measured.value.bands[1].std == 0.0
    assert measured.value.correlations == ((1.0, None), (None, None))


def test_pixel_moments_match_independent_concatenated_reference() -> None:
    rng = np.random.default_rng(31)
    images = tuple(
        rng.random((3, height, width), dtype=np.float32)
        for height, width in (
            (2, 3),
            (3, 4),
            (1, 7),
        )
    )
    expected = np.concatenate([image.reshape(3, -1).astype(np.float64) for image in images], axis=1)
    measured = measure_pixels(images, ("BLUE", "GREEN", "RED"), histogram_bins=8)
    assert isinstance(measured, Ok)
    result = measured.value
    assert result.n_pixels == 25
    np.testing.assert_allclose(
        [band.mean for band in result.bands], expected.mean(axis=1), rtol=1e-14
    )
    np.testing.assert_allclose(
        [band.std for band in result.bands], expected.std(axis=1), rtol=1e-14
    )
    np.testing.assert_allclose(
        np.asarray(result.correlations, dtype=np.float64), np.corrcoef(expected), atol=1e-14
    )
    for index, band in enumerate(result.bands):
        counts, _ = np.histogram(expected[index], bins=result.histogram_bin_edges)
        assert band.histogram_counts == tuple(counts)
        assert sum(band.histogram_counts) == result.n_pixels


def test_pixel_inputs_require_finite_unit_float32_and_exact_band_layout() -> None:
    image = np.zeros((1, 2, 3), dtype=np.float32)
    assert isinstance(measure_pixels((), ("RED",)), Err)
    assert isinstance(measure_pixels((image.astype(np.float64),), ("RED",)), Err)
    assert isinstance(measure_pixels((image,), ("RED", "RED")), Err)
    assert isinstance(measure_pixels((image + np.float32(2),), ("RED",)), Err)
    assert isinstance(measure_pixels((image + np.float32(np.nan),), ("RED",)), Err)
