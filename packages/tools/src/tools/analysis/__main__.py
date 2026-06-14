"""Module entry point: ``python -m tools.analysis ...`` dispatches to the CLI.

Satisfies: REQ-OBS-SIL-001.
"""

from __future__ import annotations

# stdlib
import sys

# internal
from tools.analysis.cli import main

if __name__ == "__main__":
    sys.exit(main())
