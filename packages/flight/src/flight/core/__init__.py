"""Compute / C&DH host (composition root, config loading, scheduling).

This phase populates only config loading; the composition root, scheduler, bus
router wiring, storage, telemetry aggregator, and FDIR coordinator are added as
the subsystem apps come online.
"""

from flight.core.config_loader import load_config

__all__ = ["load_config"]
