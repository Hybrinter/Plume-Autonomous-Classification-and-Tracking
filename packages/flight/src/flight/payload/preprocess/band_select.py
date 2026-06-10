"""flight.payload.preprocess.band_select -- reorder demosaicked band planes for inference.

Satisfies: REQ-AIML-PREP-001, REQ-AIML-IMAG-001

After CFA separation the band planes arrive in SensorConfig.mosaic_layout (row-major
2x2 cell) order. select_bands reorders them into the InferenceConfig.input_bands order
the model expects. The band vocabulary is BLUE/GREEN/RED/NIR (the 2x2 mosaic filter
passbands), which approximate Sentinel-2 B2 (~490 nm) / B3 (~560 nm) / B4 (~665 nm) /
B8 (~842 nm) so Sentinel-2-derived training data remains a valid domain (spec Section 2).

This module is layout-agnostic: it only matches names, it does not assume any fixed
index, so the legacy fixed BAND_INDICES table is gone.

Contains:
  - select_bands: gather/reorder layout-ordered planes into the requested band order,
    returning Err(FRAME_MALFORMED) on a plane-count or unknown-name mismatch.
"""

from __future__ import annotations

# third-party
import numpy as np

# internal
from flight.libs.types import Err, FaultCode, Ok, Result


def select_bands(
    planes: np.ndarray,  # np.ndarray[float32, (4, H, W)], in mosaic_layout cell order
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
    try:
        indices = [layout.index(name) for name in band_names]
    except ValueError:
        return Err(FaultCode.FRAME_MALFORMED)
    return Ok(planes[indices, :, :])  # np.ndarray[float32, (len(band_names), H, W)]
