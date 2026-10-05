"""Env-driven HAL driver selection for the composition roots.

select_drivers maps a PactConfig.drivers axis vector to a concrete Drivers
bundle. It lives in flight.core (a composition root), so it is the one place
besides flight.core.main and sim.sil permitted to import BOTH driver sets --
allowed by the drivers-from-composition-roots-only import contract (flight.core is
not a source of that contract). Real-driver SDK modules are imported lazily, only
inside the 'real' branch they back, so importing this module never requires an SDK.
The HAL Protocols (flight.hal.interfaces) and the DetectorBackend Protocol
(flight.payload.inference) are pure-Protocol and SDK-free, so they are imported at module
top to statically type each branch local; that is what removes any need for a cast or
type: ignore at the Drivers(...) construction.

The clock axis is NOT acted on here: the composition root selects RealClock vs
ManualClock from config.drivers.clock BEFORE calling this function and passes
the chosen Clock in.

Contains:
  - SimDriverInputs: the sim-only construction inputs (frames, detector, packets, readings).
  - select_drivers: resolve each axis to a sim stand-in or a real driver.

Satisfies: REQ-OPER-HIGH-002 (the validated driver config selects deployment axes).
"""

from __future__ import annotations

# stdlib
from dataclasses import dataclass

# internal
from flight.core.composition import Drivers
from flight.hal.drivers_sim import (
    SimGimbal,
    SimIssEphemeris,
    SimScalarSensor,
    SimSensor,
    SimStationLink,
)
from flight.hal.interfaces import (
    GimbalActuator,
    ImagingSensor,
    IssEphemeris,
    ScalarSensor,
    StationLink,
)
from flight.libs.config import PactConfig
from flight.libs.time import Clock
from flight.libs.types import MosaicFrame
from flight.payload.inference import InferenceRuntime, OnnxRuntimeFactory, ScriptedDetector


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


def select_drivers(
    config: PactConfig,
    clock: Clock,
    sim_inputs: SimDriverInputs | None = None,
) -> Drivers:
    """Resolve the driver axis vector to a concrete Drivers bundle.

    Per-axis rules (from config.drivers):
      - sensor: 'sim' -> SimSensor(frames); 'real' -> RealSensor(clock). The real
        camera gets NO startup exposure/gain here: the payload's graph-owned
        imaging policy applies settings through its own stop/apply/start
        handling once a policy activates.
      - thermal_sensor + power_sensor follow the sensor axis: 'sim' ->
        SimScalarSensor(readings); 'real' -> RealScalarSensor().
      - gimbal: 'sim' -> SimGimbal(clock, cfg); 'real' -> RealGimbal(clock, cfg).
      - ephemeris: 'sim' -> SimIssEphemeris(clock, cfg); 'real' -> RealIssEphemeris().
      - compute: 'sim' -> an explicit scripted InferenceRuntime over the passed
        ScriptedDetector; 'real' -> an EMPTY InferenceRuntime carrying the lazy
        OnnxRuntimeFactory. Nothing reads model files or constructs an ONNX
        session here; the INIT lifecycle loads and verifies a session before
        the runtime can serve detection.
      - link: 'sim' -> SimStationLink(inbound_packets); 'real' -> RealStationLink(cfg, clock).

    Args:
        config: The validated PactConfig (provides the driver axes + per-driver config).
        clock: The Clock already chosen by the root from config.drivers.clock.
        sim_inputs: The sim construction inputs; required when any selected axis is 'sim'.

    Returns:
        A Drivers bundle with each axis resolved to a sim stand-in or a real driver.

    Raises:
        ValueError: If any selected axis is 'sim' but sim_inputs is None. Real
            model artifacts no longer fail here: the lazy factory reports load
            failures as typed Results during the INIT lifecycle.

    Notes:
        Real driver SDK modules (PySpin/socket) are imported lazily inside
        their 'real' branches, so this module imports SDK-free. flight.core.main
        and sim.sil are the only other places allowed to construct drivers. Each branch
        local is typed with its HAL Protocol, so the Drivers(...) construction type-checks
        with no cast or type: ignore.
    """
    env = config.drivers

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

        sensor = RealSensor(clock=clock)
        thermal_sensor = RealScalarSensor()
        power_sensor = RealScalarSensor()

    # --- gimbal ---
    gimbal: GimbalActuator
    if env.gimbal == "sim":
        _require_inputs()
        gimbal = SimGimbal(
            clock=clock,
            cfg=config.gimbal,
            inner_dt_s=config.controller.inner.dt_s,
        )
    else:
        from flight.hal.drivers_real import RealGimbal

        gimbal = RealGimbal(clock=clock, cfg=config.gimbal)

    # --- ephemeris ---
    ephemeris: IssEphemeris
    if env.ephemeris == "sim":
        ephemeris = SimIssEphemeris(clock=clock, cfg=config.ephemeris)
    else:
        from flight.hal.drivers_real import RealIssEphemeris

        ephemeris = RealIssEphemeris()

    # --- compute (lazy inference runtime) ---
    inference: InferenceRuntime
    if env.compute == "sim":
        inference = InferenceRuntime.from_scripted(_require_inputs().detector)
    else:
        inference = InferenceRuntime(factory=OnnxRuntimeFactory(config))

    # --- link (station transport) ---
    station: StationLink
    if env.link == "sim":
        station = SimStationLink(_require_inputs().inbound_packets)
    else:
        from flight.hal.drivers_real import RealStationLink

        station = RealStationLink(cfg=config.link, clock=clock)

    return Drivers(
        sensor=sensor,
        gimbal=gimbal,
        ephemeris=ephemeris,
        inference=inference,
        station=station,
        thermal_sensor=thermal_sensor,
        power_sensor=power_sensor,
    )
