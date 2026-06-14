"""Storage service interfaces (Protocols): the two faces of the core data store.

The core-hosted StorageService (flight.core.storage) is file-backed, checksummed, and quota'd.
Apps depend only on these Protocols; the composition root injects the concrete service. The two
faces are split so each consumer sees only what it needs (interface segregation):

  - StorageWriter: injected into the payload app so large science products (mask thumbnails)
    are persisted by a direct call, bypassing the bus (the large-artifact invariant).
  - StorageReader: injected into iss_iface so it can fetch a stored product's bytes at
    downlink transmission time, given only the compact storage entry id carried on the bus.

These live in flight.hal.interfaces (the injected-abstraction layer) so payload and iss_iface
import them without a layering violation; the concrete service lives in the composition root.

Satisfies: REQ-DATA-STORE-001.
"""

from __future__ import annotations

# stdlib
from typing import Protocol, runtime_checkable

# internal
from flight.libs.types import DownlinkPriority, FaultCode, Result


@runtime_checkable
class StorageWriter(Protocol):
    """Write face of the data store: persist a checksummed, quota'd entry by direct call."""

    def store(
        self, item_id: str, data: bytes, priority: DownlinkPriority
    ) -> Result[str, FaultCode]:
        """Persist data under a human-readable item_id and return its storage entry id.

        Args:
            item_id: A human-readable product/record identifier (used in the on-disk name).
            data: The raw bytes to persist (checksummed on write, verified on read).
            priority: The product's downlink priority, retained so quota eviction can drop the
                lowest-priority entries first.

        Returns:
            Ok(entry_id) naming the stored entry (used later by StorageReader.read), or
            Err(FaultCode.STORAGE_FULL) when the entry cannot be admitted within the quota.
        """
        ...


@runtime_checkable
class StorageReader(Protocol):
    """Read face of the data store: fetch a stored entry's bytes, verifying its checksum."""

    def read(self, entry_id: str) -> Result[bytes, FaultCode]:
        """Read back the bytes of a stored entry, verifying the stored checksum.

        Args:
            entry_id: The entry id returned by StorageWriter.store.

        Returns:
            Ok(data) on success, or Err(FaultCode) if the entry is missing or its checksum
            does not match (corruption).
        """
        ...
