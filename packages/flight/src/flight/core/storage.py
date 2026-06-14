"""Core-hosted data store: checksummed, quota'd product storage + a reboot-surviving ledger.

StorageService is the single core data store (spec Section 6, Approach A). It has two faces and
one bus-consumer loop:

  - StorageWriter.store / StorageReader.read: file-backed product entries under data_root, each
    written with a SHA-256 sidecar and verified on read; admission is quota-bounded with a
    retention policy (evict the lowest-priority, then oldest, entry to make room; never evict an
    entry of higher priority than the incoming one; STORAGE_FULL + fault when it still will not
    fit). The payload uses this to persist mask thumbnails off the bus (large-artifact invariant).
  - tick(): a bus consumer that appends every TelemetryEventMsg to a telemetry log and every
    FaultEventMsg to an append-only fault ledger (JSON lines) that survives reboot (read back
    via read_fault_ledger). FDIR fault annunciation is thus durable.

No filesystem I/O happens at construction -- directories are created lazily on first write -- so
build_apps stays side-effect-free and the deterministic SIL stays hermetic (its composition root
points data_root at a temp directory). Heartbeats like every persistent-loop service.

Contains:
  - EntryMeta / StorageState: the in-memory index + counters threaded as mutable shell state.
  - StorageService: from_config(); store(); read(); tick(); run(); read_fault_ledger().

Satisfies: REQ-DATA-STORE-001, REQ-DATA-LEDGER-001.
"""

from __future__ import annotations

# stdlib
import hashlib
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path

# internal
from flight.libs.bus import MessageBus, Subscription
from flight.libs.config import FaultConfig, PactConfig, StorageConfig
from flight.libs.messages import FaultEventMsg, HeartbeatMsg, TelemetryEventMsg
from flight.libs.time import Clock
from flight.libs.types import DownlinkPriority, Err, FaultCode, MessageType, Ok, Result

SUBSYSTEM = "storage"
_LEDGER_NAME = "fault_ledger.jsonl"
_TELEMETRY_NAME = "telemetry.jsonl"
_PRODUCTS_DIR = "products"


@dataclass(slots=True, frozen=True)
class EntryMeta:
    """Index record for one stored entry.

    Fields:
        entry_id: The storage entry id (unique; the on-disk product filename).
        size: Byte length of the stored payload.
        priority: The product's downlink priority (drives retention/eviction order).
        order: Monotonic insertion counter (oldest-first tie-break for eviction).
    """

    entry_id: str
    size: int
    priority: DownlinkPriority
    order: int


@dataclass(slots=True)
class StorageState:
    """Mutable storage index owned by the shell.

    Fields:
        entries: entry_id -> EntryMeta for every live stored entry.
        total_bytes: Sum of live entry sizes (the quota is checked against this).
        next_order: Monotonic counter assigning insertion order to new entries.
        dropped_count: Number of entries evicted by the retention policy (telemetered).
    """

    entries: dict[str, EntryMeta] = field(default_factory=dict)
    total_bytes: int = 0
    next_order: int = 0
    dropped_count: int = 0


@dataclass(frozen=True)
class StorageService:
    """Core data store implementing StorageWriter + StorageReader plus a bus-consumer ledger."""

    cfg: StorageConfig
    fault_cfg: FaultConfig
    bus: MessageBus
    clock: Clock
    telemetry: Subscription[TelemetryEventMsg]
    faults: Subscription[FaultEventMsg]
    state: StorageState

    @staticmethod
    def from_config(cfg: PactConfig, bus: MessageBus, clock: Clock) -> StorageService:
        """Assemble a StorageService subscribing to telemetry + fault events.

        Args:
            cfg: Top-level PactConfig (storage for paths/quota; fault for heartbeat interval).
            bus: The shared MessageBus to subscribe to / publish onto.
            clock: Injected Clock (real or manual).

        Returns:
            A StorageService with fresh TelemetryEventMsg + FaultEventMsg subscriptions and an
            empty index. No filesystem I/O is performed here (dirs are created on first write).
        """
        return StorageService(
            cfg=cfg.storage,
            fault_cfg=cfg.fault,
            bus=bus,
            clock=clock,
            telemetry=bus.subscribe(TelemetryEventMsg),
            faults=bus.subscribe(FaultEventMsg),
            state=StorageState(),
        )

    # --- paths -----------------------------------------------------------------------------

    def _root(self) -> Path:
        """Return the data-root directory path (not created)."""
        return Path(self.cfg.data_root)

    def _products_dir(self) -> Path:
        """Return the products subdirectory path (not created)."""
        return self._root() / _PRODUCTS_DIR

    # --- StorageWriter ---------------------------------------------------------------------

    def store(
        self, item_id: str, data: bytes, priority: DownlinkPriority
    ) -> Result[str, FaultCode]:
        """Persist data under a checksummed entry, evicting lower-priority entries to fit.

        Args:
            item_id: Human-readable product identifier (embedded in the entry id).
            data: Raw bytes to persist.
            priority: Downlink priority retained for retention/eviction ordering.

        Returns:
            Ok(entry_id) on success, or Err(FaultCode.STORAGE_FULL) when the entry cannot be
            admitted within max_storage_bytes even after evicting all lower/equal-priority
            entries (also published as a STORAGE_FULL FaultEventMsg).
        """
        size = len(data)
        if size > self.cfg.max_storage_bytes:
            self._publish_fault(FaultCode.STORAGE_FULL, f"item {item_id!r} exceeds quota")
            return Err(FaultCode.STORAGE_FULL)

        self._evict_to_fit(size, priority)
        if self.state.total_bytes + size > self.cfg.max_storage_bytes:
            self._publish_fault(FaultCode.STORAGE_FULL, f"no room for {item_id!r} within quota")
            return Err(FaultCode.STORAGE_FULL)

        order = self.state.next_order
        self.state.next_order += 1
        entry_id = f"{order:08d}_{item_id}"
        products = self._products_dir()
        products.mkdir(parents=True, exist_ok=True)
        (products / entry_id).write_bytes(data)
        (products / f"{entry_id}.sha256").write_text(
            hashlib.sha256(data).hexdigest(), encoding="utf-8"
        )
        self.state.entries[entry_id] = EntryMeta(entry_id, size, priority, order)
        self.state.total_bytes += size
        return Ok(entry_id)

    # --- StorageReader ---------------------------------------------------------------------

    def read(self, entry_id: str) -> Result[bytes, FaultCode]:
        """Read a stored entry's bytes back, verifying its SHA-256 sidecar.

        Args:
            entry_id: The entry id returned by store.

        Returns:
            Ok(data) when the entry exists and its checksum matches, else
            Err(FaultCode.STORAGE_CORRUPT) (missing entry, missing/mismatched checksum, or read
            error).
        """
        if entry_id not in self.state.entries:
            return Err(FaultCode.STORAGE_CORRUPT)
        products = self._products_dir()
        path = products / entry_id
        sidecar = products / f"{entry_id}.sha256"
        try:
            data = path.read_bytes()
            expected = sidecar.read_text(encoding="utf-8").strip()
        except OSError:
            return Err(FaultCode.STORAGE_CORRUPT)
        if hashlib.sha256(data).hexdigest() != expected:
            return Err(FaultCode.STORAGE_CORRUPT)
        return Ok(data)

    # --- bus consumer ----------------------------------------------------------------------

    def tick(self) -> None:
        """Persist pending telemetry to the telemetry log and faults to the fault ledger."""
        root = self._root()
        telemetry_lines = []
        while not self.telemetry.empty():
            event = self.telemetry.get_nowait()
            telemetry_lines.append(
                json.dumps(
                    {
                        "ts": event.timestamp_utc,
                        "subsystem": event.subsystem,
                        "event": event.event_name,
                        "payload": event.payload,
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
        fault_lines = []
        while not self.faults.empty():
            fault = self.faults.get_nowait()
            fault_lines.append(
                json.dumps(
                    {
                        "ts": fault.timestamp_utc,
                        "fault_code": fault.fault_code.value,
                        "subsystem": fault.subsystem,
                        "detail": fault.detail,
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
        if telemetry_lines:
            self._append_lines(root / _TELEMETRY_NAME, telemetry_lines)
        if fault_lines:
            self._append_lines(root / _LEDGER_NAME, fault_lines)

    def read_fault_ledger(self) -> list[dict[str, object]]:
        """Read the reboot-surviving fault ledger back as a list of records (oldest first).

        Returns:
            The parsed ledger records, or an empty list if the ledger does not yet exist.
        """
        path = self._root() / _LEDGER_NAME
        if not path.exists():
            return []
        records: list[dict[str, object]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
        return records

    def run(self, stop_event: threading.Event) -> None:
        """Run the persistence loop until stop_event is set, emitting periodic heartbeats.

        Args:
            stop_event: threading.Event; the loop exits cleanly once it is set.
        """
        sequence = 0
        last_heartbeat = self.clock.monotonic_s()
        while not stop_event.is_set():
            self.tick()
            now = self.clock.monotonic_s()
            if now - last_heartbeat >= self.fault_cfg.watchdog_interval_s:
                self.bus.publish(
                    HeartbeatMsg(
                        msg_type=MessageType.HEARTBEAT,
                        timestamp_utc=self.clock.wall_clock_iso(),
                        subsystem=SUBSYSTEM,
                        sequence=sequence,
                    )
                )
                sequence += 1
                last_heartbeat = now
            stop_event.wait(timeout=self.fault_cfg.watchdog_interval_s)

    # --- helpers ---------------------------------------------------------------------------

    def _evict_to_fit(self, needed: int, incoming: DownlinkPriority) -> None:
        """Evict lowest-priority, then oldest, entries until needed bytes fit (retention).

        Never evicts an entry of higher priority than the incoming item: stops once only
        higher-priority entries remain (the caller then returns STORAGE_FULL). Each eviction
        deletes the product file + sidecar and increments dropped_count.
        """
        while self.state.total_bytes + needed > self.cfg.max_storage_bytes:
            evictable = [
                meta
                for meta in self.state.entries.values()
                if meta.priority.value >= incoming.value
            ]
            if not evictable:
                return
            victim = max(evictable, key=lambda m: (m.priority.value, -m.order))
            products = self._products_dir()
            (products / victim.entry_id).unlink(missing_ok=True)
            (products / f"{victim.entry_id}.sha256").unlink(missing_ok=True)
            del self.state.entries[victim.entry_id]
            self.state.total_bytes -= victim.size
            self.state.dropped_count += 1

    @staticmethod
    def _append_lines(path: Path, lines: list[str]) -> None:
        """Append JSON-line records to a log file, creating the parent directory lazily."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for line in lines:
                handle.write(line + "\n")

    def _publish_fault(self, code: FaultCode, detail: str) -> None:
        """Publish a FaultEventMsg from the storage subsystem onto the bus."""
        self.bus.publish(
            FaultEventMsg(
                msg_type=MessageType.FAULT_EVENT,
                timestamp_utc=self.clock.wall_clock_iso(),
                fault_code=code,
                subsystem=SUBSYSTEM,
                detail=detail,
            )
        )
