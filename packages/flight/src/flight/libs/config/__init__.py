"""Typed, frozen flight configuration dataclasses.

Each subsystem receives its own sub-config. Defaults here MUST match
config/default.toml (enforced by tests/test_config_defaults.py).
"""

from flight.libs.config.config import (
    CommandIngressConfig,
    CommsConfig,
    ControllerConfig,
    FaultConfig,
    GimbalConfig,
    InferenceConfig,
    LinkConfig,
    PactConfig,
    PreprocessingConfig,
    SensorConfig,
    StorageConfig,
)

__all__ = [
    "CommsConfig",
    "CommandIngressConfig",
    "ControllerConfig",
    "FaultConfig",
    "GimbalConfig",
    "InferenceConfig",
    "LinkConfig",
    "PactConfig",
    "PreprocessingConfig",
    "SensorConfig",
    "StorageConfig",
]
