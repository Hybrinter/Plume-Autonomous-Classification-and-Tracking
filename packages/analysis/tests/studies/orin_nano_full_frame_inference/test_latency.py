"""Formula and locked-constant tests for the Orin Super inference study.

These tests do not time a GPU. They lock area scale, FLOP scale, duty cycle,
and the derived expected/timeout pair from eta.
"""

from __future__ import annotations

import math

from analysis.studies.orin_nano_full_frame_inference.assumptions import (
    CLS_FLOPS_TILE_G,
    ETA_COMPUTE,
    ETA_WRAP,
    FLIGHT_PRECISION,
    FRAME_H_PX,
    FRAME_W_PX,
    HAS_DLA,
    MODULE_TDP_W,
    PAYLOAD_BUS_W,
    SEG_FLOPS_TILE_G,
    TILE_H_PX,
    TILE_W_PX,
    TIMEOUT_MULT,
    area_scale,
)
from analysis.studies.orin_nano_full_frame_inference.cost import (
    cls_flops_ff_g,
    flops_full_frame_g,
    seg_flops_ff_g,
)
from analysis.studies.orin_nano_full_frame_inference.latency import (
    cls_latency,
    derive_budget,
    duty_cycle,
    mean_time_ms,
    mixed_knee_detect_ms,
    seg_latency,
)


def test_area_scale_is_exact() -> None:
    """1024x1224 over 256x256 is exactly 19.125."""
    assert FRAME_H_PX * FRAME_W_PX == 1_253_376
    assert TILE_H_PX * TILE_W_PX == 65_536
    assert area_scale() == 19.125


def test_flop_scale_is_area_times_tile() -> None:
    """Full-frame G counts are the locked 256 G counts times area scale."""
    assert cls_flops_ff_g() == flops_full_frame_g(CLS_FLOPS_TILE_G)
    assert seg_flops_ff_g() == flops_full_frame_g(SEG_FLOPS_TILE_G)
    assert cls_flops_ff_g() == CLS_FLOPS_TILE_G * 19.125
    assert seg_flops_ff_g() == SEG_FLOPS_TILE_G * 19.125


def test_duty_cycle_search_is_classifier_detect_is_sum() -> None:
    """t_search is classifier wall; t_detect is classifier plus segmentor."""
    for precision in ("fp32", "fp16", "int8"):
        duty = duty_cycle(precision)
        t_cls = cls_latency(precision).wall_ms
        t_seg = seg_latency(precision).wall_ms
        assert duty.t_search_ms == t_cls
        assert duty.t_detect_ms == t_cls + t_seg


def test_wall_is_kernel_over_wrap_efficiency() -> None:
    """Wall time is kernel / ETA_WRAP with kernel = max(compute/eta, memory)."""
    lat = cls_latency("fp32")
    assert lat.kernel_ms == max(lat.compute_ms / ETA_COMPUTE, lat.memory_ms)
    assert lat.wall_ms == lat.kernel_ms / ETA_WRAP


def test_derived_budget_from_mixed_knee() -> None:
    """Expected is ceil(mixed-knee detect wall); timeout is ceil(5x expected)."""
    budget = derive_budget()
    detect = mixed_knee_detect_ms()
    assert budget.precision == "mixed"
    assert FLIGHT_PRECISION == "mixed"
    assert budget.t_detect_ms == detect
    assert budget.expected_ms == math.ceil(detect)
    assert budget.timeout_ms == math.ceil(TIMEOUT_MULT * budget.expected_ms)
    assert budget.expected_ms == 4
    assert budget.timeout_ms == 20


def test_derived_budget_matches_flight_config_defaults() -> None:
    """Dataclass defaults equal ceil(detect wall) and ceil(5x expected)."""
    from flight.libs.config import FaultConfig, InferenceConfig

    budget = derive_budget()
    assert InferenceConfig().latency_budget_ms == float(budget.expected_ms)
    assert FaultConfig().inference_timeout_ms == float(budget.timeout_ms)


def test_mean_time_at_zero_and_one() -> None:
    """p=0 is search; p=1 is detect."""
    duty = duty_cycle("fp32")
    assert mean_time_ms("fp32", 0.0) == duty.t_search_ms
    assert mean_time_ms("fp32", 1.0) == duty.t_detect_ms


def test_module_tdp_is_not_payload_bus() -> None:
    """Super module TDP is 25 W; payload-bus FDIR stays 55 W. No DLA."""
    assert MODULE_TDP_W == 25.0
    assert PAYLOAD_BUS_W == 55.0
    assert HAS_DLA is False
