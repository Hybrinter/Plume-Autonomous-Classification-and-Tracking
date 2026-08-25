"""StorageService tests: checksummed store/read, quota eviction, fault ledger."""

import dataclasses
from pathlib import Path

from flight.core.storage import StorageService
from flight.libs.bus import MessageBus
from flight.libs.config import PactConfig
from flight.libs.messages import FaultEventMsg, TelemetryEventMsg
from flight.libs.time import ManualClock
from flight.libs.types import DownlinkPriority, Err, FaultCode, MessageType, Ok


def _service(tmp_path: Path, max_bytes: int = 1_000_000) -> StorageService:
    """Build a StorageService rooted at tmp_path with the given quota."""
    cfg = PactConfig()
    cfg = dataclasses.replace(
        cfg,
        storage=dataclasses.replace(
            cfg.storage, data_root=str(tmp_path), max_storage_bytes=max_bytes
        ),
    )
    return StorageService.from_config(cfg, MessageBus(), ManualClock())


def test_store_then_read_roundtrips(tmp_path: Path) -> None:
    """A stored product reads back its exact bytes (checksum verified)."""
    svc = _service(tmp_path)
    stored = svc.store("thumb", b"hello-product", DownlinkPriority.SCIENCE_PRODUCT)
    assert isinstance(stored, Ok)
    read = svc.read(stored.value)
    assert isinstance(read, Ok)
    assert read.value == b"hello-product"


def test_read_detects_corruption(tmp_path: Path) -> None:
    """Tampering with a stored file makes read return STORAGE_CORRUPT."""
    svc = _service(tmp_path)
    stored = svc.store("thumb", b"abcde", DownlinkPriority.SCIENCE_PRODUCT)
    assert isinstance(stored, Ok)
    (tmp_path / "products" / stored.value).write_bytes(b"zzzzz")  # corrupt the payload
    read = svc.read(stored.value)
    assert isinstance(read, Err)
    assert read.error is FaultCode.STORAGE_CORRUPT


def test_quota_evicts_oldest_lowest_priority(tmp_path: Path) -> None:
    """Exceeding the quota evicts the oldest same-priority entry; dropped_count increments."""
    svc = _service(tmp_path, max_bytes=20)
    a = svc.store("a", b"0123456789", DownlinkPriority.SCIENCE_PRODUCT)  # 10 bytes
    b = svc.store("b", b"0123456789", DownlinkPriority.SCIENCE_PRODUCT)  # 10 bytes (total 20)
    assert isinstance(a, Ok) and isinstance(b, Ok)
    c = svc.store("c", b"0123456789", DownlinkPriority.SCIENCE_PRODUCT)  # evicts 'a'
    assert isinstance(c, Ok)
    assert svc.state.dropped_count == 1
    assert isinstance(svc.read(a.value), Err)  # 'a' was evicted
    assert isinstance(svc.read(c.value), Ok)


def test_oversize_item_rejected_with_fault(tmp_path: Path) -> None:
    """An item larger than the whole quota is rejected with STORAGE_FULL + a fault."""
    bus = MessageBus()
    cfg = PactConfig()
    cfg = dataclasses.replace(
        cfg, storage=dataclasses.replace(cfg.storage, data_root=str(tmp_path), max_storage_bytes=4)
    )
    svc = StorageService.from_config(cfg, bus, ManualClock())
    faults = bus.subscribe(FaultEventMsg)
    result = svc.store("big", b"way-too-large", DownlinkPriority.SCIENCE_PRODUCT)
    assert isinstance(result, Err)
    assert result.error is FaultCode.STORAGE_FULL
    assert faults.get_nowait().fault_code is FaultCode.STORAGE_FULL


def test_fault_ledger_persists_and_survives_reload(tmp_path: Path) -> None:
    """tick() appends FaultEventMsg to a ledger that a fresh service reads back (reboot-survive)."""
    bus = MessageBus()
    cfg = PactConfig()
    cfg = dataclasses.replace(
        cfg, storage=dataclasses.replace(cfg.storage, data_root=str(tmp_path))
    )
    svc = StorageService.from_config(cfg, bus, ManualClock())
    bus.publish(
        FaultEventMsg(
            msg_type=MessageType.FAULT_EVENT,
            timestamp_utc="2026-06-14T00:00:00.000Z",
            fault_code=FaultCode.THERMAL_OVER_LIMIT,
            subsystem="thermal",
            detail="too hot",
        )
    )
    svc.tick()
    # A brand-new service over the same data_root reads the persisted ledger (reboot).
    reborn = StorageService.from_config(cfg, MessageBus(), ManualClock())
    ledger = reborn.read_fault_ledger()
    assert len(ledger) == 1
    assert ledger[0]["fault_code"] == "THERMAL_OVER_LIMIT"


def test_tick_persists_telemetry(tmp_path: Path) -> None:
    """tick() appends TelemetryEventMsg to the telemetry log."""
    bus = MessageBus()
    cfg = PactConfig()
    cfg = dataclasses.replace(
        cfg, storage=dataclasses.replace(cfg.storage, data_root=str(tmp_path))
    )
    svc = StorageService.from_config(cfg, bus, ManualClock())
    bus.publish(
        TelemetryEventMsg(
            msg_type=MessageType.TELEMETRY_EVENT,
            timestamp_utc="2026-06-14T00:00:00.000Z",
            subsystem="thermal",
            event_name="thermal_sample",
            payload={"temperature_c": 25.0},
        )
    )
    svc.tick()
    assert (tmp_path / "telemetry.jsonl").exists()
