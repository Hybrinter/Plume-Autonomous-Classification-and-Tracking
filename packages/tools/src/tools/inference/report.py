"""Write figures and a markdown summary into an inference run directory.

The implementation lives in ``tools.ml_models.analysis.report``. This module
re-exports that function.

Contains:
  - write_report: emit figures/ PNGs and report.md from history and eval files.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

from tools.ml_models.analysis.report import write_report

__all__ = ["write_report"]
