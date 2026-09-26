"""Score test, export FP32 and INT8, and run the acceptance gate on a trained run.

The implementation lives in ``tools.ml_models.export.finalize``. This module
re-exports that public surface so existing callers keep one import path.

Contains:
  - FinalizeReport: paths, gate outcomes, and the test-split eval path.
  - finalize: evaluate test, export, accept, and write ``finalize.json``.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

from tools.ml_models.export.finalize import FinalizeReport, finalize

__all__ = [
    "FinalizeReport",
    "finalize",
]
