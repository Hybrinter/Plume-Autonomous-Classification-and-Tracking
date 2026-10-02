"""Typed builder protocols shared across the payload graph tests.

Each protocol fixes the exact keyword signature of the fixture factories so
call sites get full argument checking instead of ``Callable[..., ...]``.
"""

from __future__ import annotations

from typing import Protocol

from flight.libs.messages import BlobMeta, RoutedCommandMsg
from flight.payload.graphs.base import (
    EffectResult,
    InitVerificationResult,
    TickInputs,
)
from flight.payload.records import (
    ActivationKey,
    CapturedVision,
    HealthSample,
    IssSample,
)
from flight.payload.tracking import PredictorReferenceChange


class TickBuilder(Protocol):
    """Factory for one ``TickInputs``; encoder ``t_s`` defaults to ``now_s``."""

    def __call__(
        self,
        now_s: float,
        key: ActivationKey,
        *,
        encoder_angle_rad: float | None = 0.0,
        encoder_t_s: float | None = None,
        encoder_variance_rad2: float = 0.0,
        navigation: IssSample | None = None,
        vision: CapturedVision | None = None,
        health: HealthSample | None = None,
        command: RoutedCommandMsg | None = None,
        effect_results: tuple[EffectResult, ...] = (),
        verification: InitVerificationResult | None = None,
        stow_complete: bool = False,
        reference_change: PredictorReferenceChange | None = None,
    ) -> TickInputs:
        """Build one tick of observations."""
        ...


class VisionBuilder(Protocol):
    """Factory for one context-scoped ``CapturedVision``."""

    def __call__(
        self,
        t_s: float,
        key: ActivationKey,
        *,
        frame_id: str | None = None,
        blobs: tuple[BlobMeta, ...] = (),
        theta_g_rad: float | None = None,
        iss: IssSample | None = None,
        exposure_us: float = 1000.0,
        mode_flags: int = 0,
        policy_revision: int = 0,
    ) -> CapturedVision:
        """Build one captured vision packet."""
        ...


class BlobBuilder(Protocol):
    """Factory for one gate-passing ``BlobMeta``."""

    def __call__(
        self,
        blob_id: int = 1,
        centroid: tuple[float, float] = (612.0, 512.0),
        bbox: tuple[int, int, int, int] = (100, 100, 150, 150),
        pixel_area: int = 200,
    ) -> BlobMeta:
        """Build one blob."""
        ...
