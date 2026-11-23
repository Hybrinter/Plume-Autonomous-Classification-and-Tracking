"""Tests for payload.records value types relocated from control."""

import math
from dataclasses import FrozenInstanceError

import pytest
from flight.libs.types import ActivationKey
from flight.payload.records import (
    CaptureContext,
    CapturedVision,
    HealthSample,
    IssSample,
    VisionSample,
)


def _vision() -> VisionSample:
    """A minimal vision sample."""
    return VisionSample(
        t_s=1.0,
        frame_id="f1",
        z_v=0.01,
        p_cog=(100.0, 200.0),
        exposure_us=1000.0,
        blobs=(),
        mode_flags=0,
        iss=None,
    )


def test_iss_sample_fields() -> None:
    """IssSample carries ECI position, velocity, and UTC."""
    iss = IssSample(r_m=(1.0, 2.0, 3.0), v_m_s=(4.0, 5.0, 6.0), utc_s=7.0)
    assert iss.r_m == (1.0, 2.0, 3.0)
    assert iss.v_m_s == (4.0, 5.0, 6.0)
    assert iss.utc_s == 7.0


def test_vision_sample_default_theta() -> None:
    """VisionSample defaults the shutter encoder angle to None."""
    sample = _vision()
    assert sample.theta_g_rad is None
    assert sample.exposure_us == 1000.0
    angled = VisionSample(
        t_s=1.0,
        frame_id="f2",
        z_v=None,
        p_cog=None,
        exposure_us=1000.0,
        blobs=(),
        mode_flags=0,
        iss=None,
        theta_g_rad=math.radians(20.0),
    )
    assert angled.theta_g_rad == math.radians(20.0)


def test_activation_key_is_frozen_slots() -> None:
    """ActivationKey is an immutable epoch/sequence pair."""
    key = ActivationKey(epoch="e1", sequence=3)
    assert key.epoch == "e1"
    assert key.sequence == 3
    with pytest.raises(FrozenInstanceError):
        key.__setattr__("sequence", 4)


def test_capture_context_and_captured_vision() -> None:
    """CapturedVision binds a sample to its capture context."""
    context = CaptureContext(
        activation_key=ActivationKey(epoch="e1", sequence=3),
        policy_revision=2,
        model_version="v9",
    )
    captured = CapturedVision(context=context, sample=_vision())
    assert captured.context.activation_key.sequence == 3
    assert captured.context.policy_revision == 2
    assert captured.sample.frame_id == "f1"
    with pytest.raises(FrozenInstanceError):
        captured.__setattr__("context", context)


def test_health_sample_fields() -> None:
    """HealthSample carries feedback, inhibit, and containment flags."""
    health = HealthSample(feedback_valid=True, inhibit_confirmed=False, contained=True)
    assert health.feedback_valid is True
    assert health.inhibit_confirmed is False
    assert health.contained is True
