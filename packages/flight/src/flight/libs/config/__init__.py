"""Typed, frozen flight configuration dataclasses.

Each subsystem receives its own sub-config. Defaults here MUST match
config/default.toml (enforced by tests/test_config_defaults.py).
"""

from flight.libs.config.config import (
    CommsConfig,
    ControllerConfig,
    FaultConfig,
    InferenceConfig,
    PactConfig,
    PreprocessingConfig,
    SensorConfig,
    StorageConfig,
)

__all__ = [
    "CommsConfig",
    "ControllerConfig",
    "FaultConfig",
    "InferenceConfig",
    "PactConfig",
    "PreprocessingConfig",
    "SensorConfig",
    "StorageConfig",
]
