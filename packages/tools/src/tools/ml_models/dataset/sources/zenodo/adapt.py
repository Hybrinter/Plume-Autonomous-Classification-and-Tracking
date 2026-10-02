"""Optional Zenodo source adaptation before the shared dataset build."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from tools.ml_models.dataset.geometry import INPUT_BANDS
from tools.ml_models.dataset.raw import BinSpec, RawTile, RawTileRef
from tools.ml_models.dataset.sources.zenodo.annotations import rasterize_percent_mask
from tools.ml_models.dataset.sources.zenodo.archive import build_index, iter_stacks, to_native_stack
from tools.ml_models.dataset.sources.zenodo.bands import coerce_descriptions, verify_band_order
from tools.ml_models.dataset.sources.zenodo.bins import (
    DEFAULT_BINS,
    NATIVE_SIDE,
    actual_gsd,
    bin_hw,
)
from tools.ml_models.dataset.sources.zenodo.prism import WeightTable, mix_prism
from tools.ml_models.dataset.sources.zenodo.resample import resample_area


class ZenodoSource:
    """Read every labeled image once and emit all requested GSD variants.

    Class directory supplies the presence label. Annotation presence, not the
    presence label, controls segmentor eligibility. All dates and bins at one
    location share the same split group.
    """

    name = "zenodo"
    band_names = INPUT_BANDS
    domain = "unit"
    source_ref = "10.5281/zenodo.4250706"

    def __init__(
        self,
        images_tar: str | Path,
        labels_tar: str | Path,
        weight_table: WeightTable,
        bins: tuple[BinSpec, ...] = DEFAULT_BINS,
    ) -> None:
        if not bins or len({item.bin_id for item in bins}) != len(bins):
            raise ValueError("GSD bins must be nonempty with unique names")
        for item in bins:
            bin_hw(item)
        self.images_tar = Path(images_tar)
        self.weight_table = weight_table
        self.weight_table_id = weight_table.id
        self.bins = bins
        self._index = build_index(self.images_tar, Path(labels_tar))
        self._refs = tuple(
            RawTileRef(
                tile_id=f"{tile.stem}-{item.bin_id}",
                group_id=tile.location_id,
                label=float(tile.positive),
                has_mask=tile.polygons is not None,
                gsd=actual_gsd(item),
                height=bin_hw(item)[0],
                width=bin_hw(item)[1],
                frame_id=None,
                grid_rc=None,
                bin_id=item.bin_id,
            )
            for tile in self._index.tiles
            for item in bins
        )

    def index(self) -> tuple[RawTileRef, ...]:
        """Return adapted geometry rows in image-archive order."""
        return self._refs

    def iter_tiles(self) -> Iterator[RawTile]:
        """Mix native reflectance, area downsample, and rasterize the exact output grid."""
        cursor = 0
        for tile, stack, descriptions in iter_stacks(self.images_tar, self._index.tiles):
            ids = verify_band_order(coerce_descriptions(descriptions)).ids
            native = mix_prism(to_native_stack(stack), ids, self.weight_table)
            for item in self.bins:
                shape = bin_hw(item)
                image = np.clip(resample_area(native, shape), 0, 1).astype(np.float32)
                mask = (
                    None
                    if tile.polygons is None
                    else rasterize_percent_mask(
                        tile.polygons,
                        shape,
                        source_hw=(int(stack.shape[1]), int(stack.shape[2])),
                        fitted_hw=(NATIVE_SIDE, NATIVE_SIDE),
                    )
                )
                yield RawTile(self._refs[cursor], image, mask)
                cursor += 1
        if cursor != len(self._refs):
            raise ValueError("Zenodo stream ended before its index")
