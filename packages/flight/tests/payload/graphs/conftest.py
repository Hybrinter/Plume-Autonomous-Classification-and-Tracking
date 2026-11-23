"""Shared fixtures for the pure payload graph tests."""

import pytest
from flight.libs.config import PactConfig
from flight.libs.messages import BlobMeta, RoutedCommandMsg
from flight.libs.types import ActivationKey
from flight.payload.graphs.base import (
    EffectResult,
    InitVerificationResult,
    TickInputs,
)
from flight.payload.graphs.parameters import GraphParameters
from flight.payload.records import (
    CaptureContext,
    CapturedVision,
    HealthSample,
    IssSample,
    VisionSample,
)
from flight.payload.tracking import EncoderSample, PredictorReferenceChange

from . import support


@pytest.fixture
def params() -> GraphParameters:
    """Default-config graph parameters."""
    return GraphParameters(config=PactConfig())


@pytest.fixture
def key() -> ActivationKey:
    """One activation key for tick builders."""
    return ActivationKey(epoch="test", sequence=1)


@pytest.fixture
def tick() -> support.TickBuilder:
    """Build one TickInputs; encoder t_s defaults to now (fresh)."""

    def build(
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
        encoder = None
        if encoder_angle_rad is not None:
            encoder = EncoderSample(
                sample_id=f"enc:{now_s:.6f}",
                t_s=now_s if encoder_t_s is None else encoder_t_s,
                angle_rad=encoder_angle_rad,
                angle_variance_rad2=encoder_variance_rad2,
            )
        return TickInputs(
            now_s=now_s,
            timestamp_utc="2026-06-01T00:00:00.000Z",
            activation_key=key,
            encoder=encoder,
            navigation=navigation,
            vision=vision,
            health=health
            or HealthSample(feedback_valid=True, inhibit_confirmed=False, contained=False),
            command=command,
            effect_results=effect_results,
            verification=verification,
            stow_complete=stow_complete,
            reference_change=reference_change,
        )

    return build


@pytest.fixture
def vision() -> support.VisionBuilder:
    """Build one CapturedVision under a key/policy context."""

    def build(
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
        return CapturedVision(
            context=CaptureContext(
                activation_key=key,
                policy_revision=policy_revision,
                model_version="test",
            ),
            sample=VisionSample(
                t_s=t_s,
                frame_id=frame_id or f"frame:{t_s:.6f}",
                z_v=None,
                p_cog=None,
                exposure_us=exposure_us,
                blobs=blobs,
                mode_flags=mode_flags,
                iss=iss,
                theta_g_rad=theta_g_rad,
            ),
        )

    return build


@pytest.fixture
def blob() -> support.BlobBuilder:
    """Build one strong blob that passes the confidence/area gates."""

    def build(
        blob_id: int = 1,
        centroid: tuple[float, float] = (612.0, 512.0),
        bbox: tuple[int, int, int, int] = (100, 100, 150, 150),
        pixel_area: int = 200,
    ) -> BlobMeta:
        return BlobMeta(
            blob_id=blob_id,
            bbox=bbox,
            centroid_raw=centroid,
            pixel_area=pixel_area,
            mean_confidence=0.85,
            persistence_count=1,
        )

    return build
