"""Size against quality frontier over the local run catalog.

Ranking runs by a single metric answers "which is best" but not "what does
better cost", which is the question when a model has to fit a flight image. A
run is on the frontier when no other run is at least as good on the metric and
no larger. Everything else is dominated: something else matches or beats it on
both axes at once, so there is no operating point at which it is the right
choice.

Cost is measured in parameters by default, because that is what sets the
exported artifact size. FLOPs are available for the latency question instead.

Every point on one frontier must come from the same split. A catalog mixes runs
that were scored on validation alone with finalists that also carry a test
score, and reading whichever is present would rank a model against its own test
number while its rivals were read from validation. Test scores are the
pessimistic ones, so that comparison quietly penalises exactly the runs that
have been evaluated most thoroughly. The split is therefore chosen explicitly
and defaults to validation, which is the split the search is allowed to read.

A catalog also mixes stages. Stage 1 holds architecture fixed and varies the
recipe; stage 2 holds the recipe fixed and varies architecture. A frontier over
the whole catalog would treat those as the same comparison. ``run_ids``
restricts the points to one sweep JSONL, or to the union of several.
``mean_by_arch`` collapses extra seeds of the same architecture to their mean
score, so a lucky seed does not place an architecture on the frontier by itself.

Contains:
  - COST_KEYS: selectable cost axes.
  - SPLITS: selectable score splits.
  - FrontierPoint: one run reduced to metric and cost.
  - frontier_points: reduce run summaries to comparable points.
  - mean_by_arch: collapse seeds of one architecture to a mean score.
  - pareto_front: the non-dominated subset, ordered by cost.
  - knee: cheapest frontier point that still holds a baseline score.
  - knee_neighbors: the knee plus cheaper and costlier frontier neighbours.
  - score_spread: max minus min score across seeds of one architecture.
  - substitute_arch_placeholder: fill a space TOML arch placeholder.
  - orient_score: published metric to higher-is-better.
  - format_pareto: text table of a frontier.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from tools.inference.runs import load_summary

COST_KEYS: frozenset[str] = frozenset({"n_params", "flops"})

SPLITS: frozenset[str] = frozenset({"val", "test"})

_MINIMIZED_METRICS: frozenset[str] = frozenset({"bce", "brier"})

ARCH_PLACEHOLDER: str = "PLACEHOLDER_SET_FROM_STAGE_2"


@dataclass(frozen=True, slots=True)
class FrontierPoint:
    """One run reduced to the two axes of the trade-off.

    Attributes:
        run_id: Run identifier.
        arch: Architecture name.
        kind: ``classifier`` or ``segmentor``.
        score: Metric value, already oriented so higher is better.
        cost: Cost value, where lower is better.
        path: Run directory.
    """

    run_id: str
    arch: str
    kind: str
    score: float
    cost: float
    path: str


def _scalar(row: dict[str, object], *keys: str) -> float:
    """Return the first numeric value among ``keys``, or NaN."""
    for key in keys:
        item = row.get(key)
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            return float(item)
    return math.nan


def _split_score(row: dict[str, object], metric: str, split: str) -> float:
    """Return ``metric`` on ``split`` for one run summary, or NaN.

    Notes:
        ``best_val_metric`` is a valid source only when the summary names the
        same metric in ``val_metric``, since a run swept on F1 also stores that
        field and reading it as though it were mean IoU would invent a number.
    """
    score = _scalar(row, f"{split}_{metric}")
    if not math.isnan(score) or split != "val":
        return score
    if str(row.get("val_metric", "")) == metric:
        return _scalar(row, "best_val_metric")
    return math.nan


def frontier_points(
    runs: tuple[Path, ...],
    metric: str,
    cost_key: str = "n_params",
    kind: str = "",
    split: str = "val",
    run_ids: frozenset[str] | None = None,
) -> tuple[FrontierPoint, ...]:
    """Reduce run directories to comparable ``(score, cost)`` points.

    Args:
        runs: Run directories.
        metric: Metric name such as ``mean_iou`` or ``f1``.
        cost_key: ``n_params`` or ``flops``.
        kind: When set, keep only runs of that kind. Comparing a classifier
            against a segmentor is meaningless, so a mixed catalog must be
            filtered.
        split: ``val`` or ``test``. Every point is read from this split, and a
            run that does not carry it is dropped rather than read from the
            other one.
        run_ids: When set, keep only runs whose ``run_id`` is in this set.
            A stage-1 recipe sweep and a stage-2 architecture sweep share a
            catalog; mixing them would treat recipe and architecture as one
            comparison.

    Returns:
        tuple[FrontierPoint, ...]: One point per usable run. Runs missing the
        metric on ``split``, or missing the cost, are dropped.

    Raises:
        ValueError: If ``cost_key`` or ``split`` is not a known value.
    """
    if cost_key not in COST_KEYS:
        raise ValueError(f"unknown cost key {cost_key!r}")
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected val or test")
    points: list[FrontierPoint] = []
    for path in runs:
        row = load_summary(path)
        if kind and str(row.get("kind", "")) != kind:
            continue
        run_id = str(row.get("run_id", path.name))
        if run_ids is not None and run_id not in run_ids:
            continue
        score = _split_score(row, metric, split)
        cost = _scalar(row, cost_key)
        if math.isnan(score) or math.isnan(cost):
            continue
        oriented = -score if metric in _MINIMIZED_METRICS else score
        points.append(
            FrontierPoint(
                run_id=run_id,
                arch=str(row.get("arch", "")),
                kind=str(row.get("kind", "")),
                score=oriented,
                cost=cost,
                path=str(path),
            )
        )
    return tuple(points)


def mean_by_arch(points: tuple[FrontierPoint, ...]) -> tuple[FrontierPoint, ...]:
    """Collapse extra seeds of the same architecture to their mean score.

    Args:
        points: Per-run frontier points.

    Returns:
        tuple[FrontierPoint, ...]: One point per ``(kind, arch)``. Cost is the
        same for every seed of an architecture; score is the arithmetic mean.
        ``run_id`` names the architecture and the seed count. ``path`` is the
        member with the median score, so a later inspect command opens a
        typical run rather than the luckiest one.
    """
    groups: dict[tuple[str, str], list[FrontierPoint]] = {}
    for point in points:
        groups.setdefault((point.kind, point.arch), []).append(point)
    collapsed: list[FrontierPoint] = []
    for (kind, arch), members in sorted(groups.items()):
        ordered = sorted(members, key=lambda item: item.score)
        mean_score = sum(item.score for item in ordered) / float(len(ordered))
        typical = ordered[len(ordered) // 2]
        collapsed.append(
            FrontierPoint(
                run_id=f"{arch} n={len(ordered)}",
                arch=arch,
                kind=kind,
                score=mean_score,
                cost=typical.cost,
                path=typical.path,
            )
        )
    return tuple(collapsed)


def pareto_front(points: tuple[FrontierPoint, ...]) -> tuple[FrontierPoint, ...]:
    """Return the non-dominated points, ordered by increasing cost.

    Args:
        points: Candidate points from :func:`frontier_points`.

    Returns:
        tuple[FrontierPoint, ...]: Points for which no other point has both a
        score at least as high and a cost no greater. Ties on both axes keep
        the cheaper, then the first seen.
    """
    ordered = sorted(points, key=lambda item: (item.cost, -item.score))
    front: list[FrontierPoint] = []
    best = -math.inf
    for point in ordered:
        if point.score > best:
            front.append(point)
            best = point.score
    return tuple(front)


def knee(
    front: tuple[FrontierPoint, ...],
    baseline_score: float,
    spread: float = 0.0,
) -> FrontierPoint:
    """Return the cheapest frontier point that holds ``baseline_score``.

    Args:
        front: Non-dominated points, ordered by increasing cost.
        baseline_score: Oriented score (higher is better) that a point must
            hold. Pass the published metric after the same orientation
            ``frontier_points`` applies.
        spread: Allowed drop below the baseline, in oriented score units.
            A cheaper point that sits inside this band still counts as holding.

    Returns:
        FrontierPoint: The cheapest holding point.

    Raises:
        ValueError: If ``spread`` is negative, the frontier is empty, or no
            point holds the baseline.
    """
    if spread < 0.0:
        raise ValueError(f"spread must be >= 0, got {spread}")
    if not front:
        raise ValueError("empty frontier")
    floor = baseline_score - spread
    holding = tuple(point for point in front if point.score >= floor)
    if not holding:
        raise ValueError(f"no frontier point holds baseline {baseline_score} with spread {spread}")
    return holding[0]


def knee_neighbors(
    front: tuple[FrontierPoint, ...],
    selected: FrontierPoint,
    beside: int = 1,
) -> tuple[FrontierPoint, ...]:
    """Return the knee plus ``beside`` cheaper and costlier neighbours.

    Args:
        front: Non-dominated points, ordered by increasing cost.
        selected: The knee, which must be a member of ``front``.
        beside: Neighbours to keep on each side. Zero returns only the knee.

    Returns:
        tuple[FrontierPoint, ...]: A contiguous slice of ``front`` centred on
            ``selected``. Truncates at the ends of the frontier.

    Raises:
        ValueError: If ``beside`` is negative or ``selected`` is not in
            ``front``.
    """
    if beside < 0:
        raise ValueError(f"beside must be >= 0, got {beside}")
    try:
        index = front.index(selected)
    except ValueError:
        raise ValueError("knee is not a member of the frontier") from None
    lo = max(0, index - beside)
    hi = min(len(front), index + beside + 1)
    return front[lo:hi]


def score_spread(points: tuple[FrontierPoint, ...], arch: str) -> float:
    """Return max minus min score across seeds of one architecture.

    Args:
        points: Per-run points, before ``mean_by_arch``.
        arch: Architecture name.

    Returns:
        float: Score range. Zero when the architecture has fewer than two
            seeds, including when it is absent.
    """
    scores = tuple(point.score for point in points if point.arch == arch)
    if len(scores) < 2:
        return 0.0
    return max(scores) - min(scores)


def substitute_arch_placeholder(text: str, arches: tuple[str, ...]) -> str:
    """Replace ``ARCH_PLACEHOLDER`` in a space TOML with ``arches``.

    Args:
        text: Space file contents. The placeholder must appear in
            ``arch = ["PLACEHOLDER_SET_FROM_STAGE_2"]``,
            ``arch = "PLACEHOLDER_SET_FROM_STAGE_2"``, or as one quoted item
            in a longer ``arch`` list.
        arches: Architecture names, in the order they should be written.

    Returns:
        str: ``text`` with the first matching assignment or list item replaced.

    Raises:
        ValueError: If ``arches`` is empty, the placeholder is absent, a list
            assignment is missing for several names, or a scalar or in-list
            assignment is given more than one name.
    """
    if not arches:
        raise ValueError("empty architecture list")
    if ARCH_PLACEHOLDER not in text:
        raise ValueError("space text has no architecture placeholder")
    list_token = f'arch = ["{ARCH_PLACEHOLDER}"]'
    scalar_token = f'arch = "{ARCH_PLACEHOLDER}"'
    item_token = f'"{ARCH_PLACEHOLDER}"'
    quoted = ", ".join(f'"{name}"' for name in arches)
    if list_token in text:
        return text.replace(list_token, f"arch = [{quoted}]", 1)
    if scalar_token in text:
        if len(arches) != 1:
            raise ValueError("scalar arch placeholder needs exactly one architecture")
        return text.replace(scalar_token, f'arch = "{arches[0]}"', 1)
    if item_token in text:
        if len(arches) != 1:
            raise ValueError("in-list arch placeholder needs exactly one architecture")
        return text.replace(item_token, f'"{arches[0]}"', 1)
    raise ValueError("architecture placeholder is not in an arch assignment")


def orient_score(metric: str, value: float) -> float:
    """Return ``value`` oriented so that higher is better.

    Args:
        metric: Metric name. ``bce`` and ``brier`` are minimized.
        value: Published metric value.

    Returns:
        float: ``-value`` for minimized metrics, otherwise ``value``.
    """
    return -value if metric in _MINIMIZED_METRICS else value


def format_pareto(front: tuple[FrontierPoint, ...], metric: str, cost_key: str) -> str:
    """Return a text table of a frontier.

    Args:
        front: Non-dominated points.
        metric: Metric name, used for the column header.
        cost_key: Cost axis, used for the column header.

    Returns:
        str: Header plus one row per point, cheapest first.
    """
    headers = ("run_id", "kind", "arch", metric, cost_key, "path")
    lines = ["\t".join(headers)]
    for point in front:
        score = -point.score if metric in _MINIMIZED_METRICS else point.score
        lines.append(
            "\t".join(
                (
                    point.run_id,
                    point.kind,
                    point.arch,
                    f"{score:.4g}",
                    f"{point.cost:.6g}",
                    point.path,
                )
            )
        )
    return "\n".join(lines) + "\n"
