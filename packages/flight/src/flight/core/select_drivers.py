"""Env-driven HAL driver selection for the composition roots.

select_drivers maps a PactConfig.environment axis vector to a concrete Drivers
bundle. It lives in flight.core (a composition root), so it is the one place
besides flight.core.main and sim.sil permitted to import BOTH driver sets --
allowed by the drivers-from-composition-roots-only import contract (flight.core is
not a source of that contract). Real-driver SDK modules are imported lazily, only
inside the 'real' branch they back, so importing this module never requires an SDK.
The HAL Protocols (flight.hal.interfaces) and the DetectorBackend Protocol
(flight.payload.model) are pure-Protocol and SDK-free, so they are imported at module
top to statically type each branch local; that is what removes any need for a cast or
type: ignore at the Drivers(...) construction.

The clock axis is NOT acted on here: the composition root selects RealClock vs
ManualClock from config.environment.clock BEFORE calling this function and passes
the chosen Clock in. The 'lock' (LaunchLock) axis does not exist (permanent VCRM gap).

Contains:
  - SimDriverInputs: the sim-only construction inputs (frames, detector, packets, readings).
  - select_drivers: resolve each axis to a sim stand-in or a real driver.

Satisfies: REQ-OPER-HIGH-002 (the validated environment config selects deployment axes).
"""

from __future__ import annotations

# stdlib
from dataclasses import dataclass

# internal
from flight.core.composition import Drivers
from flight.hal.drivers_sim import (
    SimGimbal,
    SimLaunchLock,
    SimScalarSensor,
    SimSensor,
    SimStationLink,
)
from flight.hal.interfaces import (
    GimbalActuator,
    ImagingSensor,
    ScalarSensor,
    StationLink,
)
from flight.libs.config import PactConfig
from flight.libs.time import Clock
from flight.libs.types import LaunchLockState, MosaicFrame, Ok
from flight.payload.model import DetectorBackend, ScriptedDetector


@dataclass(frozen=True, slots=True)
class SimDriverInputs:
    """The sim-only inputs the in-process drivers replay.

    These are supplied by the SIL/GSE composition root when one or more axes are
    'sim'. Fields are consumed only by the sim branches of select_drivers; the real
    branches ignore them.
    """

    frames: list[MosaicFrame]  # raw mosaic frames the SimSensor replays
    detector: ScriptedDetector  # scripted detector reused when compute axis is 'sim'
    inbound_packets: list[bytes]  # CCSDS TC packets the SimStationLink delivers
    thermal_readings: list[float]  # temperature readings (Celsius) for the thermal sensor
    power_readings: list[float]  # power readings (Watts) for the electrical sensor
    launch_lock_engaged: bool = (
        False  # SimLaunchLock initial state; False -> RELEASED (ops default)
    )


def select_drivers(
    config: PactConfig,
    clock: Clock,
    sim_inputs: SimDriverInputs | None = None,
) -> Drivers:
    """Resolve the environment axis vector to a concrete Drivers bundle.

    Per-axis rules (from config.environment):
      - sensor: 'sim' -> SimSensor(frames); 'real' -> RealSensor(clock) then command
        the configured startup exposure/gain (SystemExit on Err -- an unusable camera
        at startup is unrecoverable).
      - thermal_sensor + power_sensor follow the sensor axis: 'sim' ->
        SimScalarSensor(readings); 'real' -> RealScalarSensor().
      - gimbal: 'sim' -> SimGimbal(clock, cfg); 'real' -> RealGimbal(clock, cfg).
      - compute: 'sim' -> the passed ScriptedDetector; 'real' -> OnnxDetector(model_path).
      - link: 'sim' -> SimStationLink(inbound_packets); 'real' -> RealStationLink(cfg, clock).

    Args:
        config: The validated PactConfig (provides the environment axes + per-driver config).
        clock: The Clock already chosen by the root from config.environment.clock.
        sim_inputs: The sim construction inputs; required when any selected axis is 'sim'.

    Returns:
        A Drivers bundle with each axis resolved to a sim stand-in or a real driver.

    Raises:
        ValueError: If any selected axis is 'sim' but sim_inputs is None.
        SystemExit: If the real-sensor startup exposure or gain command fails.

    Notes:
        Real driver SDK modules (PySpin/pyserial/onnxruntime/socket) are imported lazily
        inside their 'real' branches, so this module imports SDK-free. flight.core.main
        and sim.sil are the only other places allowed to construct drivers. Each branch
        local is typed with its HAL Protocol, so the Drivers(...) construction type-checks
        with no cast or type: ignore.
    """
    env = config.environment

    def _require_inputs() -> SimDriverInputs:
        """Return sim_inputs or raise: a 'sim' axis demands construction inputs."""
        if sim_inputs is None:
            raise ValueError("select_drivers requires sim_inputs when any axis is 'sim'")
        return sim_inputs

    # --- sensor + the two scalar sensors (they follow the sensor axis) ---
    sensor: ImagingSensor
    thermal_sensor: ScalarSensor
    power_sensor: ScalarSensor
    if env.sensor == "sim":
        inputs = _require_inputs()
        sensor = SimSensor(inputs.frames)
        thermal_sensor = SimScalarSensor(inputs.thermal_readings)
        power_sensor = SimScalarSensor(inputs.power_readings)
    else:
        from flight.hal.drivers_real import RealScalarSensor, RealSensor

        real_sensor = RealSensor(clock=clock)
        exposure_result = real_sensor.set_exposure_us(config.sensor.default_exposure_us)
        if not isinstance(exposure_result, Ok):
            raise SystemExit(f"camera exposure setup failed: {exposure_result.error}")
        gain_result = real_sensor.set_gain_db(config.sensor.default_gain_db)
        if not isinstance(gain_result, Ok):
            raise SystemExit(f"camera gain setup failed: {gain_result.error}")
        sensor = real_sensor
        thermal_sensor = RealScalarSensor()
        power_sensor = RealScalarSensor()

    # --- gimbal ---
    gimbal: GimbalActuator
    if env.gimbal == "sim":
        _require_inputs()
        gimbal = SimGimbal(clock=clock, cfg=config.gimbal)
    else:
        from flight.hal.drivers_real import RealGimbal

        gimbal = RealGimbal(clock=clock, cfg=config.gimbal)

    # --- compute (detector backend) ---
    detector: DetectorBackend
    if env.compute == "sim":
        detector = _require_inputs().detector
    else:
        from flight.payload.model import OnnxDetector

        detector = OnnxDetector(
            config.inference.model_path,
            latency_budget_ms=config.inference.latency_budget_ms,
        )

    # --- link (station transport) ---
    station: StationLink
    if env.link == "sim":
        station = SimStationLink(_require_inputs().inbound_packets)
    else:
        from flight.hal.drivers_real import RealStationLink

        station = RealStationLink(cfg=config.link, clock=clock)

    # The launch lock has no real driver yet (hardware-deferred, a permanent VCRM gap), so
    # every profile -- including all-"real" flight -- wires the SimLaunchLock stand-in. Flight
    # (sim_inputs=None) starts ENGAGED (launch configuration); a SIL run starts from its
    # sim_inputs flag (RELEASED by default, the operational config, so pointing SIL runs move).
    lock_engaged = sim_inputs.launch_lock_engaged if sim_inputs is not None else True
    launch_lock = SimLaunchLock(
        LaunchLockState.ENGAGED if lock_engaged else LaunchLockState.RELEASED
    )
    return Drivers(
        sensor=sensor,
        gimbal=gimbal,
        detector=detector,
        station=station,
        thermal_sensor=thermal_sensor,
        power_sensor=power_sensor,
        launch_lock=launch_lock,
    )
