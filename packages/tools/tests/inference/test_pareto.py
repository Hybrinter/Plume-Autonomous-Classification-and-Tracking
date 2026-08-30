"""Tests for size-versus-quality Pareto frontier over run directories."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tools.inference.pareto import (
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


def _write_run(
    root: Path,
    name: str,
    *,
    kind: str = "segmentor",
    arch: str = "unet",
    n_params: int = 1000,
    flops: int = 2000,
    **metrics: float | str,
) -> Path:
    """Write a minimal summary.json run directory."""
    dest = root / name
    dest.mkdir()
    payload: dict[str, object] = {
        "run_id": name,
        "kind": kind,
        "arch": arch,
        "n_params": n_params,
        "flops": flops,
    }
    payload.update(metrics)
    (dest / "summary.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return dest


def test_frontier_points_reads_only_the_requested_split(tmp_path: Path) -> None:
    """A run carrying both splits is read from the requested one, never the other."""
    both = _write_run(tmp_path, "both", test_mean_iou=0.9, val_mean_iou=0.5)

    on_val = frontier_points((both,), "mean_iou", split="val")
    on_test = frontier_points((both,), "mean_iou", split="test")

    assert on_val[0].score == pytest.approx(0.5)
    assert on_test[0].score == pytest.approx(0.9)


def test_frontier_points_drops_runs_missing_the_requested_split(tmp_path: Path) -> None:
    """A val-only run is absent from a test frontier rather than read from val.

    Mixing splits would rank a test-scored finalist against validation-scored
    rivals, which flatters the rivals because test scores are the pessimistic
    ones.
    """
    val_only = _write_run(tmp_path, "val-only", val_mean_iou=0.7)
    scored = _write_run(tmp_path, "scored", test_mean_iou=0.6, val_mean_iou=0.8)

    ids = {point.run_id for point in frontier_points((val_only, scored), "mean_iou", split="test")}

    assert ids == {"scored"}


def test_frontier_points_uses_best_val_metric_only_for_a_matching_metric(
    tmp_path: Path,
) -> None:
    """best_val_metric stands in for the val score only when it names that metric."""
    matching = _write_run(tmp_path, "matching", val_metric="mean_iou", best_val_metric=0.4)
    mismatched = _write_run(tmp_path, "mismatched", val_metric="f1", best_val_metric=0.95)

    points = frontier_points((matching, mismatched), "mean_iou", split="val")

    by_id = {point.run_id: point for point in points}
    assert by_id["matching"].score == pytest.approx(0.4)
    assert "mismatched" not in by_id


def test_frontier_points_rejects_an_unknown_split(tmp_path: Path) -> None:
    """An unknown split name is refused rather than silently treated as val."""
    run = _write_run(tmp_path, "run", val_mean_iou=0.5)

    with pytest.raises(ValueError, match="unknown split"):
        frontier_points((run,), "mean_iou", split="train")


def test_frontier_points_filters_by_kind_and_drops_missing_fields(tmp_path: Path) -> None:
    """Kind filter applies and runs without metric or cost are omitted."""
    segmentor = _write_run(tmp_path, "seg", kind="segmentor", val_mean_iou=0.5)
    classifier = _write_run(tmp_path, "clf", kind="classifier", val_mean_iou=0.8)
    no_metric = _write_run(tmp_path, "no-metric", kind="segmentor")
    no_cost = tmp_path / "no-cost"
    no_cost.mkdir()
    (no_cost / "summary.json").write_text(
        json.dumps({"run_id": "no-cost", "kind": "segmentor", "val_mean_iou": 0.3}) + "\n",
        encoding="utf-8",
    )

    points = frontier_points(
        (segmentor, classifier, no_metric, no_cost),
        "mean_iou",
        kind="segmentor",
    )
    assert [point.run_id for point in points] == ["seg"]


def test_frontier_points_negates_minimized_metrics(tmp_path: Path) -> None:
    """bce and brier scores are negated so higher is better."""
    bce_run = _write_run(tmp_path, "bce", val_bce=0.2)
    brier_run = _write_run(tmp_path, "brier", val_brier=0.3)

    bce_point = frontier_points((bce_run,), "bce")[0]
    brier_point = frontier_points((brier_run,), "brier")[0]
    assert bce_point.score == pytest.approx(-0.2)
    assert brier_point.score == pytest.approx(-0.3)


def test_frontier_points_rejects_unknown_cost_key(tmp_path: Path) -> None:
    """An unknown cost axis raises ValueError."""
    run = _write_run(tmp_path, "run", val_mean_iou=0.5)
    with pytest.raises(ValueError, match="unknown cost key"):
        frontier_points((run,), "mean_iou", cost_key="latency")


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


def test_format_pareto_table_and_metric_orientation(tmp_path: Path) -> None:
    """format_pareto prints a header, rows cheapest first, and restores bce sign."""
    run = _write_run(tmp_path, "run-a", kind="segmentor", arch="unet", val_bce=0.25)
    point = frontier_points((run,), "bce")[0]
    front = pareto_front((point,))
    table = format_pareto(front, "bce", "n_params")

    lines = table.splitlines()
    assert lines[0] == "run_id\tkind\tarch\tbce\tn_params\tpath"
    assert lines[1].startswith("run-a\tsegmentor\tunet\t0.25\t1000\t")
    assert table.endswith("\n")


def test_frontier_points_filters_by_run_ids(tmp_path: Path) -> None:
    """run_ids keeps one sweep's trials and drops catalog neighbours."""
    kept = _write_run(tmp_path, "stage2", arch="unet_w16", val_mean_iou=0.55)
    other = _write_run(tmp_path, "stage1", arch="unet_w32", val_mean_iou=0.60)

    points = frontier_points((kept, other), "mean_iou", run_ids=frozenset({"stage2"}))

    assert {point.run_id for point in points} == {"stage2"}


def test_mean_by_arch_averages_seeds_and_picks_a_typical_path(tmp_path: Path) -> None:
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
