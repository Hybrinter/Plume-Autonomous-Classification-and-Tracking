"""Packaging metadata for the vendored Xeryon v1.88 library.

The upstream ``Xeryon.py`` module is intentionally not imported here: it opens
the pyserial dependency at module import time.  Flight code must import it
lazily from its hardware driver after validating the deployment settings.
"""

__version__ = "1.88.0"
UPSTREAM_LIBRARY_VERSION = "v1.88"

__all__ = ["UPSTREAM_LIBRARY_VERSION", "__version__"]
