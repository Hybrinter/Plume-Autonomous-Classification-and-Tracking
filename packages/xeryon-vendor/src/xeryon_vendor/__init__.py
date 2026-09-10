"""Packaging metadata for the vendored Xeryon v1.88 library.

The upstream ``Xeryon.py`` module is intentionally not re-exported here. The
real gimbal driver imports ``xeryon_vendor.Xeryon`` at module load; serial I/O
still waits until the controller ``start()`` call.
"""

__version__ = "1.88.0"
UPSTREAM_LIBRARY_VERSION = "v1.88"

__all__ = ["UPSTREAM_LIBRARY_VERSION", "__version__"]
