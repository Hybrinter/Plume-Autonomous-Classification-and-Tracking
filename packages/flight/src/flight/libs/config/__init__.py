"""Typed, frozen flight configuration dataclasses.

Each subsystem receives its own sub-config. Defaults here MUST match
config/default.toml (enforced by tests/test_config_defaults.py).
"""

from flight.libs.config.config import (
    ArbiterConfig,
    AxisMode,
    CommandIngressConfig,
    CommandRouterConfig,
    CommsConfig,
    ControllerConfig,
    EnvironmentConfig,
    EphemerisConfig,
    FaultConfig,
    GimbalConfig,
    InferenceConfig,
    InnerLoopConfig,
    LinkConfig,
    OuterLoopConfig,
    PactConfig,
    PositionLoopConfig,
    PreprocessingConfig,
    ResidualConfig,
    SensorConfig,
    StorageConfig,
    ThermalConfig,
    VisionConfig,
)

__all__ = [
    "ArbiterConfig",
    "AxisMode",
    "CommandIngressConfig",
    "CommandRouterConfig",
    "CommsConfig",
    "ControllerConfig",
    "EnvironmentConfig",
    "EphemerisConfig",
    "FaultConfig",
    "GimbalConfig",
    "InferenceConfig",
    "InnerLoopConfig",
    "LinkConfig",
    "OuterLoopConfig",
    "PactConfig",
    "PositionLoopConfig",
    "PreprocessingConfig",
    "ResidualConfig",
    "SensorConfig",
    "StorageConfig",
    "ThermalConfig",
    "VisionConfig",
]
