"""Size against quality frontier over the local run catalog.

The implementation lives in ``tools.ml_models.analysis.pareto``. This module
re-exports that catalog. ``substitute_arch_placeholder`` stays available for
callers. The inference CLI does not rewrite a sweep file from ``--write-space``.

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
  - flight_pareto: parameter count against full-frame hit rate.
  - study_pareto: parameter count against val Dice.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

from tools.ml_models.analysis.pareto import (
    ARCH_PLACEHOLDER,
    COST_KEYS,
    FULL_FRAME_HIT_RATE,
    SPLITS,
    STUDY_DICE_METRIC,
    FrontierPoint,
    flight_pareto,
    format_pareto,
    frontier_points,
    has_full_frame_hit_rate,
    knee,
    knee_neighbors,
    mean_by_arch,
    orient_score,
    pareto_front,
    partition_catalog,
    score_spread,
    study_pareto,
    substitute_arch_placeholder,
)

__all__ = [
    "ARCH_PLACEHOLDER",
    "COST_KEYS",
    "FULL_FRAME_HIT_RATE",
    "SPLITS",
    "STUDY_DICE_METRIC",
    "FrontierPoint",
    "flight_pareto",
    "format_pareto",
    "frontier_points",
    "has_full_frame_hit_rate",
    "knee",
    "knee_neighbors",
    "mean_by_arch",
    "orient_score",
    "pareto_front",
    "partition_catalog",
    "score_spread",
    "study_pareto",
    "substitute_arch_placeholder",
]
