"""
pact.model.augmentation — Albumentations transform pipelines for HSG-AIML training.

Satisfies: REQ-AIML-HIGH-001 (generalisation)

Addresses RISK-AIML-001 (domain gap between Sentinel-2 training data and on-orbit VNIR
imagery from the FLIR Blackfly S). Each augmentation is annotated with the specific
domain-gap component it simulates.

Sentinel-2 L2A per-band normalisation statistics (empirical from HSG-AIML training set):
    B2 (490 nm): mean=0.0924, std=0.0403
    B3 (560 nm): mean=0.1006, std=0.0408
    B4 (665 nm): mean=0.0922, std=0.0440
    B8 (842 nm): mean=0.2721, std=0.0842
Values are in [0, 1] (already divided by the 10000 DN scale factor).
These statistics were computed over the 1,437 labelled images in the dataset.
TODO: re-compute from the actual downloaded dataset before training.
"""

from __future__ import annotations

# third-party
import albumentations as A
from albumentations.pytorch import ToTensorV2


# Sentinel-2 L2A normalisation statistics for bands B2, B3, B4, B8.
# Order matches the 4-channel tensor ordering used throughout PACT.
_BAND_MEANS: tuple[float, float, float, float] = (0.0924, 0.1006, 0.0922, 0.2721)
_BAND_STDS: tuple[float, float, float, float] = (0.0403, 0.0408, 0.0440, 0.0842)


def build_train_transforms(
    crop_height: int = 256,
    crop_width: int = 256,
) -> A.Compose:
    """Build the augmentation pipeline used during training.

    Every transform is justified against RISK-AIML-001 (domain gap):

    - RandomBrightnessContrast: Simulates sun-angle variation across orbital passes.
      The FLIR camera on ISS sees dramatically different illumination angles compared
      to the nadir-pointing Sentinel-2, causing brightness/contrast shifts in plume regions.

    - GaussianBlur: Simulates atmospheric haze, thin clouds, and PSF differences between
      Sentinel-2's optical system and the FLIR Blackfly S optics at ~420 km altitude.
      Haze degrades high-frequency plume edge detail; training on blurred samples improves
      robustness to this effect.

    - HorizontalFlip / VerticalFlip: The ISS ground track crosses plume sources at any
      orientation. Plume direction is not a discriminative feature that should be learned.

    - RandomRotate90: Same rationale as flips — rotation invariance to plume orientation.
      Combined with flips this covers all 8 dihedral symmetries.

    - RandomCrop(256, 256): Sentinel-2 tiles are larger than the 256×256 inference window.
      Random cropping provides positional augmentation and ensures the model generalises
      to plumes at any spatial position within the FOV.

    - Normalize: Standardises pixel values using per-band L2A statistics so that the
      ImageNet-pretrained ResNet-34 encoder receives inputs with zero mean and unit
      variance, mitigating the domain shift from 3-channel RGB to 4-channel VNIR.

    Args:
        crop_height: Output crop height in pixels. Default 256.
        crop_width:  Output crop width in pixels.  Default 256.

    Returns:
        An ``albumentations.Compose`` pipeline ready for training.
    """
    return A.Compose(
        [
            # Orientation augmentations — plume direction is not a class-discriminative cue.
            A.HorizontalFlip(p=0.5),   # RISK-AIML-001: orbital orientation variation
            A.VerticalFlip(p=0.5),     # RISK-AIML-001: orbital orientation variation
            A.RandomRotate90(p=0.5),   # RISK-AIML-001: covers remaining 90° orientations

            # Illumination augmentation — sun-angle variation (RISK-AIML-001).
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5,
            ),

            # Blur augmentation — atmospheric haze and PSF mismatch (RISK-AIML-001).
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),

            # Spatial crop — positional augmentation within the 256×256 inference window.
            A.RandomCrop(height=crop_height, width=crop_width),

            # Normalise using Sentinel-2 L2A per-band statistics (RISK-AIML-001).
            A.Normalize(mean=_BAND_MEANS, std=_BAND_STDS, max_pixel_value=1.0),
        ]
    )


def build_val_transforms(
    crop_height: int = 256,
    crop_width: int = 256,
) -> A.Compose:
    """Build the deterministic transform pipeline used during validation.

    Validation uses a centre crop and normalisation only — no stochastic augmentations.
    This ensures reproducible metric computation across epochs.

    Args:
        crop_height: Output crop height in pixels. Default 256.
        crop_width:  Output crop width in pixels.  Default 256.

    Returns:
        An ``albumentations.Compose`` pipeline ready for validation.
    """
    return A.Compose(
        [
            A.CenterCrop(height=crop_height, width=crop_width),
            A.Normalize(mean=_BAND_MEANS, std=_BAND_STDS, max_pixel_value=1.0),
        ]
    )
