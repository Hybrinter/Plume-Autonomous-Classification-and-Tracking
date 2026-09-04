"""Headroom and max-upload size tests for the Orin Super inference study."""

from __future__ import annotations

from pathlib import Path

from analysis.studies.orin_nano_full_frame_inference.assumptions import (
    AGX_ORIN_32GB,
    AGX_ORIN_64GB,
    CATALOG,
    CLS_FLIGHT_ONNX_BYTES,
    CLS_ONNX,
    CPU_NAME,
    MAX_DAILY_UPLINK_BYTES,
    MAX_FRAME_RATE_HZ,
    MAX_STORAGE_BYTES,
    MAX_UPLINK_BPS,
    NANO_SUPER,
    PAYLOAD_BUS_W,
    SEG_FLIGHT_ONNX_BYTES,
    SEG_ONNX,
    catalog_area_scale,
)
from analysis.studies.orin_nano_full_frame_inference.headroom import (
    binding_usable_ceiling,
    board_compare,
    board_compares,
    catalog_fit,
    catalog_fits,
    factory_scaled_pair_bytes,
    max_params_for_wall,
    memory_floor_wall_ms,
    pair_onnx_bytes,
    size_ceilings,
    tensor_working_set,
    utilization,
)
from analysis.studies.orin_nano_full_frame_inference.latency import (
    mixed_knee_detect_ms,
    seg_latency,
)
from flight.libs.config import CommsConfig, SensorConfig, StorageConfig


def test_catalog_area_scale_is_exact() -> None:
    """1024x1224 over 128x128 is exactly 76.5."""
    assert catalog_area_scale() == 76.5


def test_flight_onnx_bytes_match_factory_files() -> None:
    """Locked shipped sizes match data/models/active_*.onnx."""
    cls_path = Path(CLS_ONNX)
    seg_path = Path(SEG_ONNX)
    assert cls_path.is_file()
    assert seg_path.is_file()
    assert cls_path.stat().st_size == CLS_FLIGHT_ONNX_BYTES
    assert seg_path.stat().st_size == SEG_FLIGHT_ONNX_BYTES


def test_comms_and_sensor_caps_match_flight_defaults() -> None:
    """Uplink, storage, and frame-rate locks match config dataclasses."""
    assert CommsConfig().max_daily_uplink_bytes == MAX_DAILY_UPLINK_BYTES
    assert CommsConfig().max_uplink_rate_bps == MAX_UPLINK_BPS
    assert StorageConfig().max_storage_bytes == MAX_STORAGE_BYTES
    assert SensorConfig().max_frame_rate_hz == MAX_FRAME_RATE_HZ


def test_shipped_pair_is_under_one_mib() -> None:
    """Factory ONNX pair is well under a megabyte."""
    assert pair_onnx_bytes() == float(CLS_FLIGHT_ONNX_BYTES + SEG_FLIGHT_ONNX_BYTES)
    assert pair_onnx_bytes() < 1024.0 * 1024.0


def test_utilization_expected_is_tight_timeout_and_frame_are_not() -> None:
    """Detect wall uses most of 4 ms expected and little of 20 ms / 35 Hz."""
    occ = utilization()
    assert occ.expected_ms == 4
    assert occ.timeout_ms == 20
    assert 0.80 < occ.frac_expected < 1.0
    assert occ.frac_timeout < 0.20
    assert occ.frac_frame < 0.15
    assert occ.gpu_busy_frac_35hz < 0.10
    assert occ.dram_free_gb > 3.0


def test_memory_floor_is_just_under_shipped_walls() -> None:
    """I/O plus maps already cost ~1.6 ms wall; shipped nets sit on that floor."""
    floor = memory_floor_wall_ms()
    detect = mixed_knee_detect_ms()
    assert 1.4 < floor < 1.7
    assert detect > 2.0 * floor
    assert detect < 2.2 * floor


def test_timeout_scaled_factory_pair_is_a_few_mib() -> None:
    """Growing the factory family to 20 ms stays near 5 MiB, far under uplink."""
    timeout_bytes = factory_scaled_pair_bytes(20.0)
    assert 4.0 * 1024 * 1024 < timeout_bytes < 6.0 * 1024 * 1024
    assert timeout_bytes * 10 < float(MAX_DAILY_UPLINK_BYTES)


def test_binding_usable_ceiling_is_the_timeout_not_uplink() -> None:
    """Real-time uploads bind on FDIR timeout, not the 100 MiB daily cap."""
    usable = binding_usable_ceiling()
    ceilings = {row.name: row for row in size_ceilings()}
    assert usable.binds == "latency"
    assert usable.name == "20 ms FDIR timeout"
    assert usable.max_pair_bytes < ceilings["daily uplink"].max_pair_bytes
    assert usable.max_pair_bytes < ceilings["DRAM loadable"].max_pair_bytes
    assert ceilings["4 ms expected"].max_pair_bytes < usable.max_pair_bytes


def test_efficientnet_b0_fits_timeout_not_expected() -> None:
    """EfficientNet-B0 at full frame fits 20 ms with the shipped segmentor."""
    rows = {row.arch.name: row for row in catalog_fits()}
    b0 = rows["efficientnet_b0_pt"]
    assert b0.arch.kind == "cls"
    assert b0.fits_timeout is True
    assert b0.fits_expected is False
    assert b0.fits_frame is True


def test_resnet18_misses_timeout() -> None:
    """ResNet-18 at 1024x1224 misses the 20 ms FDIR timeout."""
    rows = {row.arch.name: row for row in catalog_fits()}
    resnet = rows["resnet18_pt"]
    assert resnet.fits_timeout is False
    assert resnet.fits_frame is False
    assert resnet.pair_detect_ms > 20.0


def test_unet_w32_sep_fits_timeout_baseline_unet_does_not() -> None:
    """Compact U-Net-sep fits 20 ms; the 13 M baseline U-Net does not."""
    rows = {row.arch.name: row for row in catalog_fits()}
    compact = rows["unet_w32_sep"]
    baseline = rows["unet_baseline"]
    assert compact.fits_timeout is True
    assert compact.fits_expected is False
    assert baseline.fits_timeout is False
    assert baseline.pair_detect_ms > 100.0


def test_catalog_fit_uses_shipped_partner() -> None:
    """A catalog classifier pair time is that wall plus shipped segmentor INT8."""
    shuf = next(item for item in CATALOG if item.name == "shufflenetv2_x0_5_pt")
    fit = catalog_fit(shuf)
    assert fit.pair_detect_ms == fit.wall_ms + seg_latency("int8").wall_ms


def test_max_params_zero_below_memory_floor() -> None:
    """A wall under the I/O floor cannot hold any parameters."""
    floor = memory_floor_wall_ms()
    assert max_params_for_wall(floor * 0.5, "fp16", "cls") == 0.0
    assert max_params_for_wall(10.0, "fp16", "cls") > 343_000.0


def test_both_model_tensors_are_under_100_mib() -> None:
    """Weights plus I/O plus maps for both nets stay well under 100 MiB."""
    mem = tensor_working_set()
    mib = 1024.0 * 1024.0
    assert mem.cls_weight_bytes + mem.seg_weight_bytes < mib
    assert mem.tensors_bytes < 100.0 * mib
    assert mem.frac_dram_tensors < 0.02
    assert mem.frac_dram_gpu_held < 0.20


def test_jetson_has_no_discrete_vram_pool() -> None:
    """Nano Super DRAM is the unified 8 GB LPDDR5 pool."""
    assert NANO_SUPER.dram_gb == 8.0
    assert CPU_NAME.startswith("Arm Cortex-A78AE")
    assert NANO_SUPER.dla_count == 0


def test_agx_cpu_is_same_a78ae_with_more_cores() -> None:
    """AGX is 8 or 12 Cortex-A78AE cores, about 1.7x or 2.6x core-GHz."""
    nano = board_compare(NANO_SUPER)
    agx32 = board_compare(AGX_ORIN_32GB)
    agx64 = board_compare(AGX_ORIN_64GB)
    assert nano.cpu_x == 1.0
    assert AGX_ORIN_32GB.cpu_cores == 8
    assert AGX_ORIN_64GB.cpu_cores == 12
    assert NANO_SUPER.cpu_cores == 6
    assert 1.6 < agx32.cpu_x < 1.9
    assert 2.4 < agx64.cpu_x < 2.8
    assert agx64.gpu_x < 3.0
    assert agx64.dram_x == 8.0


def test_agx_tdp_exceeds_payload_bus_nano_does_not() -> None:
    """AGX 60 W is over the 55 W payload-bus FDIR; Nano 25 W is not."""
    assert board_compare(NANO_SUPER).tdp_vs_payload < 0.5
    assert board_compare(AGX_ORIN_64GB).tdp_vs_payload > 1.0
    assert AGX_ORIN_64GB.tdp_w > PAYLOAD_BUS_W
    assert NANO_SUPER.tdp_w < PAYLOAD_BUS_W


def test_latency_unconstrained_bind_is_uplink_not_dram() -> None:
    """With time ignored, daily uplink is tighter than loadable DRAM."""
    ceilings = {row.name: row for row in size_ceilings()}
    assert ceilings["daily uplink"].max_pair_bytes < ceilings["DRAM loadable"].max_pair_bytes
    assert board_compares()[0].spec.name == NANO_SUPER.name
