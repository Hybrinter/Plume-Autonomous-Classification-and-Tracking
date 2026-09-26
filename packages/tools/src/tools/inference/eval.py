"""Held-out split evaluation for a trained run directory.

The implementation lives in ``tools.ml_models.analysis.eval``. This module
re-exports ``evaluate``.

Contains:
  - evaluate: score a checkpoint on train, val, or test and write eval.json.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

from tools.ml_models.analysis.eval import evaluate

__all__ = ["evaluate"]
