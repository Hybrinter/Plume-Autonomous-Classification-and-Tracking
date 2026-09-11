#!/usr/bin/env python3
"""Download the Mommert smoke-plume corpus (Zenodo 4250706).

Default run prints the citation and local checksum status. It does not download.
Pass ``--download`` to fetch missing files into data/raw/.
"""

from __future__ import annotations

from tools.inference.fetch import main

if __name__ == "__main__":
    raise SystemExit(main())
