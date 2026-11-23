#!/usr/bin/env python3
"""Run the band-matrix GSD sweep over the Zenodo 4250706 corpus.

Thin wrapper for ``python -m tools.ml_models.studies.band_matrix``.
"""

from __future__ import annotations

from tools.ml_models.studies.band_matrix.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
