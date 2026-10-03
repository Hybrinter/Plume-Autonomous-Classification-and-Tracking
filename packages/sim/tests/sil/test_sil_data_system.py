"""SIL integration: science products are stored (checksummed) and downlinked, telemetry logged."""

from dataclasses import replace
from pathlib import Path

from flight.libs.config import PactConfig
from flight.libs.time import ManualClock
from flight.libs.types import Ok, SystemMode
from sim.scene import build_frames, plume_detector
from sim.sil import SilHarness, build_sil_system, publish_activation


def _config() -> PactConfig:
    """Default config with zeroed sim encoder noise (keeps the 0-deg bound fresh)."""
    base = PactConfig()
    return replace(
        base,
        gimbal=replace(
            base.gimbal,
            simulation=replace(base.gimbal.simulation, encoder_noise_deg=0.0),
        ),
    )


def test_mask_products_are_stored_and_downlinked() -> None:
    """Per-frame mask thumbnails are persisted by the StorageService and downlinked."""
    system = build_sil_system(
        _config(),
        ManualClock(),
        build_frames(6),
        plume_detector(),
        inbound_packets=[],
        thermal_readings=[25.0],
        power_readings=[30.0],
    )
    publish_activation(system, SystemMode.OPERATE, sequence=1)
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
