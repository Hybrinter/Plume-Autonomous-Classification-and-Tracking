"""Backend-agnostic blob extraction from a segmentation probability mask.

Connected-component analysis turning a (H, W) probability map into discrete
BlobMeta detections. Lifted verbatim from the original InferenceEngine so the
scripted and ONNX detector backends share identical detection geometry.
"""

import numpy as np
import scipy.ndimage

from flight.libs.messages import BlobMeta


def extract_blobs(
    prob_mask: np.ndarray,
    confidence_gate: float,
    min_blob_area_px: int,
) -> tuple[BlobMeta, ...]:
    """Extract connected-component blobs from a confidence mask.

    Args:
        prob_mask: (H, W) float32 probability map in [0, 1].
        confidence_gate: Threshold at/above which a pixel counts as positive.
        min_blob_area_px: Minimum pixel count for a blob to be reported.

    Returns:
        Blobs with blob_id and persistence_count set to 0 (assigned later by the tracker).
    """
    binary_mask = (prob_mask >= confidence_gate).astype(np.uint8)
    labeled, num_features = scipy.ndimage.label(binary_mask)

    blobs: list[BlobMeta] = []
    for label_idx in range(1, num_features + 1):
        component = labeled == label_idx
        pixel_area = int(component.sum())
        if pixel_area < min_blob_area_px:
            continue
        ys, xs = np.where(component)
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        cx = float(xs.mean())
        cy = float(ys.mean())
        mean_conf = float(prob_mask[component].mean())
        blobs.append(
            BlobMeta(
                blob_id=0,
                bbox=(x_min, y_min, x_max, y_max),
                centroid_raw=(cx, cy),
                pixel_area=pixel_area,
                mean_confidence=mean_conf,
                persistence_count=0,
            )
        )
    return tuple(blobs)
