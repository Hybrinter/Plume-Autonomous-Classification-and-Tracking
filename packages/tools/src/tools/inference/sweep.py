"""Cartesian hyperparameter sweep over the local run catalog.

Public names are defined in ``tools.ml_models.train.sweep``.

Contains:
  - load_sweep_space: expand a space file into TrainConfig trials.
  - completed_run_ids: run identifiers already present in a JSONL.
  - sweep: run trials and write sweep.jsonl.
"""

from __future__ import annotations

from tools.ml_models.train.sweep import completed_run_ids, load_sweep_space, sweep

__all__ = ["completed_run_ids", "load_sweep_space", "sweep"]
