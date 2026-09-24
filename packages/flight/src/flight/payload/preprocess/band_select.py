"""flight.payload.preprocess.band_select -- reorder demosaicked band planes for inference.

Satisfies: REQ-AIML-PREP-001, REQ-AIML-IMAG-001

After CFA separation the band planes arrive in SensorConfig.mosaic_layout (row-major
2x2 cell) order. select_bands reorders them into the InferenceConfig.input_bands order
the model expects. The band vocabulary is BLUE/GREEN/RED/NIR: the 2x2 CFA cell passbands
behind a quad-band front filter centred at ~490 / ~560 / ~665 / ~842 nm, matching
Sentinel-2 B2 / B3 / B4 / B8 so Sentinel-2-derived training data remains a valid domain.

This module is layout-agnostic: it only matches names, it does not assume any fixed
index. band_index is the single place a band NAME is resolved to a channel INDEX; the
quality heuristics use it so that a reordered input_bands never silently changes which
plane they read.

Contains:
  - band_index: resolve one band name to its channel index in a band-name tuple,
    returning Err(FRAME_MALFORMED) when the name is absent.
  - select_bands: gather/reorder layout-ordered planes into the requested band order,
    returning Err(FRAME_MALFORMED) on a plane-count or unknown-name mismatch.
"""

from __future__ import annotations

# third-party
import numpy as np

# internal
from flight.libs.types import Err, FaultCode, Ok, Result


def band_index(band_names: tuple[str, ...], name: str) -> Result[int, FaultCode]:
    """Resolve a band name to its channel index within band_names.

    Inputs:
        band_names (tuple[str, ...]): Channel order of a (C, H, W) array, e.g.
            InferenceConfig.input_bands.
        name (str): Band name to look up, e.g. "NIR".

    Outputs:
        Result[int, FaultCode]:
            Ok(int) -- the channel index of name in band_names;
            Err(FaultCode.FRAME_MALFORMED) -- name is not present.
    """
    try:
        return Ok(band_names.index(name))
    except ValueError:
        return Err(FaultCode.FRAME_MALFORMED)


def select_bands(
    planes: np.ndarray,  # np.ndarray[float32, (len(layout), H, W)], in mosaic_layout cell order
    layout: tuple[str, ...],
    band_names: tuple[str, ...],
) -> Result[np.ndarray, FaultCode]:
    """Reorder demosaicked band planes from layout order into band_names order.

    Inputs:
        planes (np.ndarray[float32, (len(layout), H, W)]): Band planes in
            SensorConfig.mosaic_layout (row-major 2x2 cell) order.
        layout (tuple[str, ...]): The band name of each plane, e.g.
            ("BLUE", "GREEN", "RED", "NIR").
        band_names (tuple[str, ...]): Requested output order
            (InferenceConfig.input_bands).

    Outputs:
        Result[np.ndarray, FaultCode]:
            Ok(np.ndarray[float32, (len(band_names), H, W)]) with channels in
            band_names order;
            Err(FaultCode.FRAME_MALFORMED) if a requested name is absent from layout,
            if planes is not 3-D, or if the plane count disagrees with layout.

    Notes:
        Pure gather by integer index; no copy of pixel data beyond numpy's fancy-index
        result. The output channel order follows band_names exactly, not layout order.
    """
    if planes.ndim != 3 or planes.shape[0] != len(layout):
        return Err(FaultCode.FRAME_MALFORMED)
    indices: list[int] = []
    for name in band_names:
        idx = band_index(layout, name)
        if isinstance(idx, Err):
            return Err(idx.error)
        indices.append(idx.value)
    return Ok(planes[indices, :, :])  # np.ndarray[float32, (len(band_names), H, W)]
