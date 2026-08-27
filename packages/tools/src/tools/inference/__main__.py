"""Module entry point for ``python -m tools.inference``.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

from tools.inference.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
