"""Tests for size-versus-quality Pareto frontier over run directories."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from flight.libs.types import Err
from tools.ml_models.analysis.pareto import (
    FrontierPoint,
    format_pareto,
    frontier_points,
    knee,
    knee_neighbors,
    mean_by_arch,
    orient_score,
    pareto_front,
    score_spread,
    substitute_arch_placeholder,
)


def _write_run(root: Path, name: str, **fields: object) -> Path:
    """Write a minimal summary.json run directory."""
    dest = root / name
    dest.mkdir()
    payload: dict[str, object] = {
        "run_id": name,
        "kind": "segmentor",
        "arch": "unet",
        "n_params": 1000,
        "flops": 2000,
    }
    payload.update(fields)
    (dest / "summary.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return dest


def test_frontier_points_fails_closed_while_catalog_is_unavailable(
    tmp_path: Path,
) -> None:
    """A readable summary.json still cannot become a frontier point."""
    run = _write_run(tmp_path, "scored", val_mean_iou=0.6)
    result = frontier_points((run,), "mean_iou", split="val")
    assert isinstance(result, Err)
    assert "unavailable" in result.error


def test_frontier_points_rejects_an_unknown_split(tmp_path: Path) -> None:
    """An unknown split name fails rather than silently treating it as val."""
    run = _write_run(tmp_path, "run", val_mean_iou=0.5)
    result = frontier_points((run,), "mean_iou", split="train")
    assert isinstance(result, Err)
    assert "unknown split" in result.error


def test_frontier_points_rejects_unknown_cost_key(tmp_path: Path) -> None:
    """An unknown cost axis returns Err."""
    run = _write_run(tmp_path, "run", val_mean_iou=0.5)
    result = frontier_points((run,), "mean_iou", cost_key="latency")
    assert isinstance(result, Err)
    assert "unknown cost key" in result.error


def test_pareto_front_excludes_dominated_and_orders_by_cost() -> None:
    """Dominated points drop out and survivors sort by increasing cost."""
    dominated = FrontierPoint("dom", "unet", "segmentor", 0.6, 120.0, "/dom")
    cheap = FrontierPoint("cheap", "unet", "segmentor", 0.7, 80.0, "/cheap")
    mid = FrontierPoint("mid", "unet", "segmentor", 0.8, 100.0, "/mid")
    large = FrontierPoint("large", "unet", "segmentor", 0.9, 150.0, "/large")

    front = pareto_front((dominated, cheap, mid, large))
    assert [point.run_id for point in front] == ["cheap", "mid", "large"]
    assert [point.cost for point in front] == [80.0, 100.0, 150.0]


def test_pareto_front_tie_keeps_cheaper_then_first_seen() -> None:
    """Equal score and cost keep the cheaper point, then the first seen."""
    first = FrontierPoint("first", "unet", "segmentor", 0.5, 100.0, "/first")
    duplicate = FrontierPoint("dup", "unet", "segmentor", 0.5, 100.0, "/dup")
    front = pareto_front((first, duplicate))
    assert [point.run_id for point in front] == ["first"]


def test_format_pareto_table_and_metric_orientation() -> None:
    """format_pareto prints a header, rows cheapest first, and restores bce sign."""
    point = FrontierPoint("run-a", "unet", "segmentor", -0.25, 1000.0, "/run-a")
    front = pareto_front((point,))
    table = format_pareto(front, "bce", "n_params")

    lines = table.splitlines()
    assert lines[0] == "run_id\tkind\tarch\tbce\tn_params\tpath"
    assert lines[1].startswith("run-a\tsegmentor\tunet\t0.25\t1000\t")
    assert table.endswith("\n")


def test_mean_by_arch_averages_seeds_and_picks_a_typical_path() -> None:
    """Seeds of one architecture collapse to their mean score."""
    low = FrontierPoint("a-0", "unet_w16", "segmentor", 0.40, 100.0, "/low")
    mid = FrontierPoint("a-1", "unet_w16", "segmentor", 0.50, 100.0, "/mid")
    high = FrontierPoint("a-2", "unet_w16", "segmentor", 0.60, 100.0, "/high")
    other = FrontierPoint("b-0", "unet_w8", "segmentor", 0.45, 50.0, "/other")

    collapsed = mean_by_arch((low, mid, high, other))
    by_arch = {point.arch: point for point in collapsed}

    assert by_arch["unet_w16"].score == pytest.approx(0.50)
    assert by_arch["unet_w16"].run_id == "unet_w16 n=3"
    assert by_arch["unet_w16"].path == "/mid"
    assert by_arch["unet_w8"].score == pytest.approx(0.45)
    assert by_arch["unet_w8"].run_id == "unet_w8 n=1"


def test_knee_is_cheapest_point_that_holds_the_baseline() -> None:
    """The knee is the cheapest frontier point at or above the baseline."""
    cheap = FrontierPoint("cheap", "unet_w8", "segmentor", 0.50, 80.0, "/cheap")
    mid = FrontierPoint("mid", "unet_w16", "segmentor", 0.56, 100.0, "/mid")
    large = FrontierPoint("large", "unet", "segmentor", 0.60, 150.0, "/large")
    front = (cheap, mid, large)

    selected = knee(front, baseline_score=0.55)

    assert selected.run_id == "mid"


def test_knee_takes_a_cheaper_neighbour_inside_the_spread() -> None:
    """A cheaper point that sits inside the allowed drop is the knee."""
    cheap = FrontierPoint("cheap", "unet_w8", "segmentor", 0.548, 80.0, "/cheap")
    mid = FrontierPoint("mid", "unet_w16", "segmentor", 0.56, 100.0, "/mid")
    front = (cheap, mid)

    selected = knee(front, baseline_score=0.55, spread=0.01)

    assert selected.run_id == "cheap"


def test_knee_rejects_empty_front_negative_spread_and_no_holder() -> None:
    """Empty, negative spread, and a front below the baseline all raise."""
    cheap = FrontierPoint("cheap", "unet_w8", "segmentor", 0.40, 80.0, "/cheap")

    with pytest.raises(ValueError, match="empty frontier"):
        knee((), baseline_score=0.55)
    with pytest.raises(ValueError, match="spread must be"):
        knee((cheap,), baseline_score=0.55, spread=-0.01)
    with pytest.raises(ValueError, match="no frontier point holds"):
        knee((cheap,), baseline_score=0.55)


def test_knee_neighbors_returns_a_contiguous_slice() -> None:
    """Neighbours are the frontier slice centred on the knee, truncated at the ends."""
    cheap = FrontierPoint("cheap", "unet_w8", "segmentor", 0.50, 80.0, "/cheap")
    mid = FrontierPoint("mid", "unet_w16", "segmentor", 0.56, 100.0, "/mid")
    large = FrontierPoint("large", "unet", "segmentor", 0.60, 150.0, "/large")
    front = (cheap, mid, large)

    around_mid = knee_neighbors(front, mid, beside=1)
    around_cheap = knee_neighbors(front, cheap, beside=1)
    only_mid = knee_neighbors(front, mid, beside=0)

    assert [point.run_id for point in around_mid] == ["cheap", "mid", "large"]
    assert [point.run_id for point in around_cheap] == ["cheap", "mid"]
    assert [point.run_id for point in only_mid] == ["mid"]


def test_knee_neighbors_rejects_a_stranger_and_a_negative_beside() -> None:
    """A point that is not on the frontier, or a negative beside, is refused."""
    cheap = FrontierPoint("cheap", "unet_w8", "segmentor", 0.50, 80.0, "/cheap")
    stranger = FrontierPoint("other", "unet", "segmentor", 0.60, 150.0, "/other")

    with pytest.raises(ValueError, match="not a member"):
        knee_neighbors((cheap,), stranger, beside=1)
    with pytest.raises(ValueError, match="beside must be"):
        knee_neighbors((cheap,), cheap, beside=-1)


def test_score_spread_is_range_of_seeds_and_zero_without_a_pair() -> None:
    """Spread is max minus min for an architecture; absent or singleton is zero."""
    low = FrontierPoint("a-0", "unet_w16", "segmentor", 0.40, 100.0, "/low")
    high = FrontierPoint("a-1", "unet_w16", "segmentor", 0.60, 100.0, "/high")
    other = FrontierPoint("b-0", "unet_w8", "segmentor", 0.45, 50.0, "/other")

    assert score_spread((low, high, other), "unet_w16") == pytest.approx(0.20)
    assert score_spread((low, high, other), "unet_w8") == pytest.approx(0.0)
    assert score_spread((low, high, other), "missing") == pytest.approx(0.0)


def test_orient_score_negates_minimized_metrics() -> None:
    """bce and brier flip sign; other metrics pass through."""
    assert orient_score("f1", 0.9) == pytest.approx(0.9)
    assert orient_score("bce", 0.2) == pytest.approx(-0.2)


def test_substitute_arch_placeholder_fills_list_and_scalar() -> None:
    """List placeholders take several names; a scalar placeholder takes one."""
    listed = 'kind = "segmentor"\narch = ["PLACEHOLDER_SET_FROM_STAGE_2"]\n'
    scalar = 'kind = "classifier"\narch = "PLACEHOLDER_SET_FROM_STAGE_2"\n'

    filled_list = substitute_arch_placeholder(listed, ("unet_w16", "unet_w32"))
    filled_scalar = substitute_arch_placeholder(scalar, ("resnet18_pt",))

    assert filled_list == 'kind = "segmentor"\narch = ["unet_w16", "unet_w32"]\n'
    assert filled_scalar == 'kind = "classifier"\narch = "resnet18_pt"\n'


def test_substitute_arch_placeholder_rejects_empty_missing_and_scalar_mismatch() -> None:
    """Empty names, a missing placeholder, and a multi-name scalar all raise."""
    listed = 'arch = ["PLACEHOLDER_SET_FROM_STAGE_2"]\n'
    scalar = 'arch = "PLACEHOLDER_SET_FROM_STAGE_2"\n'
    bare = "arch = PLACEHOLDER_SET_FROM_STAGE_2\n"

    with pytest.raises(ValueError, match="empty architecture list"):
        substitute_arch_placeholder(listed, ())
    with pytest.raises(ValueError, match="no architecture placeholder"):
        substitute_arch_placeholder('arch = ["unet"]\n', ("unet_w16",))
    with pytest.raises(ValueError, match="exactly one architecture"):
        substitute_arch_placeholder(scalar, ("resnet18", "resnet34"))
    with pytest.raises(ValueError, match="not in an arch assignment"):
        substitute_arch_placeholder(bare, ("unet_w16",))


def test_substitute_arch_placeholder_replaces_an_item_in_a_longer_list() -> None:
    """A placeholder that sits beside a real name is replaced in place."""
    text = 'arch = [\n  "unet",\n  "PLACEHOLDER_SET_FROM_STAGE_2",\n]\n'

    filled = substitute_arch_placeholder(text, ("unet_w16",))

    assert filled == 'arch = [\n  "unet",\n  "unet_w16",\n]\n'


def test_substitute_arch_placeholder_rejects_several_names_for_an_in_list_item() -> None:
    """An in-list placeholder cannot expand to several architectures."""
    text = 'arch = [\n  "unet",\n  "PLACEHOLDER_SET_FROM_STAGE_2",\n]\n'

    with pytest.raises(ValueError, match="in-list arch placeholder"):
        substitute_arch_placeholder(text, ("unet_w16", "unet_w32"))
