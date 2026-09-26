"""Empty result tables for the band study.

The implementation lives in ``tools.ml_models.analysis.results``. This module
re-exports the table writers.

Contains:
  - write_stub_tables: markdown rows with a blank score column.
  - write_filled_tables: native and ground-sample scores when present.
"""

from __future__ import annotations

from tools.ml_models.analysis.results import write_filled_tables, write_stub_tables

__all__ = ["write_filled_tables", "write_stub_tables"]
