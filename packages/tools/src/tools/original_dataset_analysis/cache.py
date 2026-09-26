"""Memmap of native GeoTIFF stacks from one archive pass.

Public names are defined in ``tools.ml_models.data.zenodo``.

Contains:
  - TileCache: stem-addressed float32 stacks.
  - to_native_stack: pad or crop a near-native tile to 120.
  - build_cache: write the memmap from ``iter_stacks``.
  - build_mask_cache: rasterize each annotated tile once.
  - open_cache: reopen a cache written by ``build_cache``.
  - SidePack: one prepared ground-sample size.
  - prepare_side: resample the native cache once.
  - open_side_pack: open a prepared side pack.
"""

from __future__ import annotations

from tools.ml_models.data.zenodo import (
    SidePack as SidePack,
)
from tools.ml_models.data.zenodo import (
    TileCache as TileCache,
)
from tools.ml_models.data.zenodo import (
    build_cache as build_cache,
)
from tools.ml_models.data.zenodo import (
    build_mask_cache as build_mask_cache,
)
from tools.ml_models.data.zenodo import (
    open_cache as open_cache,
)
from tools.ml_models.data.zenodo import (
    open_side_pack as open_side_pack,
)
from tools.ml_models.data.zenodo import (
    prepare_side as prepare_side,
)
from tools.ml_models.data.zenodo import (
    to_native_stack as to_native_stack,
)

__all__ = [
    "SidePack",
    "TileCache",
    "build_cache",
    "build_mask_cache",
    "open_cache",
    "open_side_pack",
    "prepare_side",
    "to_native_stack",
]
