"""SIL integration: science products are stored (checksummed) and downlinked, telemetry logged."""

from pathlib import Path

from flight.libs.config import PactConfig
from flight.libs.time import ManualClock
from flight.libs.types import Ok
from sim.scene import build_frames, plume_detector
from sim.sil import SilHarness, build_sil_system


def test_mask_products_are_stored_and_downlinked() -> None:
    """Per-frame mask thumbnails are persisted by the StorageService and downlinked."""
    system = build_sil_system(
        PactConfig(),
        ManualClock(),
        build_frames(6),
        plume_detector(),
        inbound_packets=[],
        thermal_readings=[25.0],
        power_readings=[30.0],
    )
    SilHarness(system).run_steps(6, dt=1.0)

    storage = system.apps.storage
    # Products were stored, one per processed frame, and read back with a verified checksum.
    assert storage.state.next_order > 0
    entry_id = next(iter(storage.state.entries))
    assert isinstance(storage.read(entry_id), Ok)

    # The downlink manager + iss_iface transmitted TM packets over the (sim) link.
    assert len(system.station.sent) > 0

    # Housekeeping telemetry was persisted to the (hermetic temp) data root.
    assert (Path(storage.cfg.data_root) / "telemetry.jsonl").exists()
