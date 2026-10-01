#!/usr/bin/env python3
"""Evaluate complete 64-tile flight frames with a conditioned model pair.

Thin wrapper for ``pact-tools ml-models frame-eval``.
"""

from __future__ import annotations

import sys

from tools.ml_models.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["frame-eval", *sys.argv[1:]]))
