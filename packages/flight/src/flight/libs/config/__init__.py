"""Typed, frozen flight configuration dataclasses.

Each subsystem receives its own sub-config. Defaults here MUST match
config/default.toml (enforced by tests/test_config_defaults.py).
"""

from flight.libs.config.config import (
    AxisMode,
    CommandIngressConfig,
    CommsConfig,
    ControllerConfig,
    EnvironmentConfig,
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
    "AxisMode",
    "CommsConfig",
    "CommandIngressConfig",
    "ControllerConfig",
    "EnvironmentConfig",
    "FaultConfig",
    "GimbalConfig",
    "InferenceConfig",
    "LinkConfig",
    "PactConfig",
    "PreprocessingConfig",
    "SensorConfig",
    "StorageConfig",
]
