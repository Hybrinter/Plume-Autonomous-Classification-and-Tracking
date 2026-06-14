"""Typed per-step signal registry for the SIL telemetry recorder.

Every observable the recorder samples each SIL step is one frozen ``Signal`` in ``REGISTRY``:
its stable dotted name, its group (one of the ten flight apps, the message ``bus``, or the
synthetic ``system`` rollup), a human title, a unit, a kind (NUMERIC or CATEGORICAL), and a pure
``extract`` callable that reads the value from a ``SampleContext``. The context bundles everything
the recorder gathered for one step -- the wired ``SilSystem`` (apps + bus + sim drivers), the
payload ``ControlState`` and FDIR watchdog entries the recorder threads through ``step_once``, the
messages drained from the recorder's own (passive, fan-out) subscriptions this step, and a one-shot
``DeviceSample`` of the sim drivers. Extractors are read-only and side-effect-free; an extractor
that raises is the recorder's signal to record NaN (numeric) or "" (categorical), so the registry
never has to defend against transient None/shape surprises.

Using a frozen ``Signal`` carrying a statically-typed ``ExtractorFn`` field (not a name->callable
dispatch map) keeps the registry strict-typed and mirrors the existing ``tools.accept`` convention
of typed ``Callable`` fields; there is no getattr-style dynamic dispatch anywhere here.

Contains:
  - SignalKind / Signal / ExtractorFn: the registry element + its value/extractor types.
  - DeviceSample / SampleContext: the per-step inputs an extractor reads.
  - REGISTRY / GROUPS: the assembled signal tuple + the ordered group names.
  - signals_for_group / signal_names: read-only registry accessors.

Satisfies: REQ-OBS-SIL-001.
"""

from __future__ import annotations

# stdlib
import enum
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

# third-party
import numpy as np

# internal
from flight.fault.watchdog import WatchdogEntry
from flight.libs.messages import (
    CommandAckMsg,
    CommandMsg,
    DownlinkItemMsg,
    FaultEventMsg,
    GimbalCommandMsg,
    HeartbeatMsg,
    InferenceResultMsg,
    LaunchLockStateMsg,
    LinkStateMsg,
    ModeChangeMsg,
    ModelDeployStateMsg,
    ModelStagedMsg,
    ProcessedFrameMsg,
    ProductRefMsg,
    RoutedCommandMsg,
    SafetyStateMsg,
    StorageWriteMsg,
    TelemetryEventMsg,
    UploadChunkMsg,
)
from flight.libs.types import DownlinkPriority, FaultCode
from flight.payload.control import ControlState
from sim.sil import SilSystem

_NAN = float("nan")

SignalValue = float | str
ExtractorFn = Callable[["SampleContext"], SignalValue]

# The 19 frozen bus message types, in a stable order (used to generate the bus signal family).
MESSAGE_TYPES: tuple[type, ...] = (
    ProcessedFrameMsg,
    InferenceResultMsg,
    GimbalCommandMsg,
    TelemetryEventMsg,
    FaultEventMsg,
    HeartbeatMsg,
    ModeChangeMsg,
    CommandMsg,
    RoutedCommandMsg,
    SafetyStateMsg,
    StorageWriteMsg,
    ProductRefMsg,
    DownlinkItemMsg,
    UploadChunkMsg,
    ModelStagedMsg,
    ModelDeployStateMsg,
    CommandAckMsg,
    LinkStateMsg,
    LaunchLockStateMsg,
)

# The nine heartbeat-emitting subsystems the FDIR watchdog monitors (mirrors MONITORED_SUBSYSTEMS).
MONITORED: tuple[str, ...] = (
    "payload",
    "iss_iface",
    "thermal",
    "electrical",
    "command_router",
    "storage",
    "downlink",
    "mechanical",
    "model_deploy",
)


class SignalKind(enum.Enum):
    """Whether a signal is a real-valued series (NUMERIC) or a label series (CATEGORICAL).

    String values mirror member names (log/serialization readability convention).
    """

    NUMERIC = "NUMERIC"
    CATEGORICAL = "CATEGORICAL"


@dataclass(frozen=True, slots=True)
class DeviceSample:
    """One-shot read of the sim HAL drivers for a single step (taken once, after step_once).

    Reading each driver once per step keeps the per-step values self-consistent and the run
    deterministic: ``read_position`` redraws seeded encoder noise on every call, so the recorder
    samples it exactly once and every gimbal extractor reads the same cached numbers.

    Fields:
        gimbal_az_meas_deg / gimbal_el_meas_deg: ``read_position`` angles (with encoder noise).
        gimbal_az_true_deg / gimbal_el_true_deg: the clean integrated pose (driver truth).
        gimbal_rate_az_deg_s / gimbal_rate_el_deg_s: the commanded RATE-mode rates (clamped).
        gimbal_mode: the active GimbalCommandMode name, or "NONE" before the first command.
        stow_switch: True once stow is commanded and the pose is within stow tolerance.
        launch_lock_state: the LaunchLockState name read from the launch-lock driver.
        link_state: the station LinkState name (AOS / LOS).
        station_sent_total: cumulative count of packets the station link has transmitted.
        sensor_index / thermal_index / power_index: replay cursors of the sim sources.
    """

    gimbal_az_meas_deg: float
    gimbal_el_meas_deg: float
    gimbal_az_true_deg: float
    gimbal_el_true_deg: float
    gimbal_rate_az_deg_s: float
    gimbal_rate_el_deg_s: float
    gimbal_mode: str
    stow_switch: bool
    launch_lock_state: str
    link_state: str
    station_sent_total: int
    sensor_index: int
    thermal_index: int
    power_index: int


@dataclass(frozen=True, slots=True)
class SampleContext:
    """Everything an extractor may read for one SIL step (assembled by the recorder).

    Fields:
        step: 1-based step index.
        t: monotonic seconds at this step (the recorder's advanced ``now``).
        system: the wired SilSystem (apps + bus + sim drivers + clock).
        payload_state: the payload ControlState the recorder threaded out of this step.
        fault_entries: the FDIR watchdog entries the recorder threaded out of this step.
        messages: messages drained from the recorder's passive subscriptions, keyed by type.
        devices: the one-shot DeviceSample for this step.
    """

    step: int
    t: float
    system: SilSystem
    payload_state: ControlState
    fault_entries: Mapping[str, WatchdogEntry]
    messages: Mapping[type, tuple[object, ...]]
    devices: DeviceSample


@dataclass(frozen=True, slots=True)
class Signal:
    """One registered per-step observable: identity + how to extract it from a SampleContext.

    Fields:
        name: stable, unique, dotted identifier (e.g. "payload.gimbal_state").
        group: the owning group ("payload", "bus", "system", ...); names the wide frame + figures.
        title: short human-readable description for plot titles and the report.
        unit: SI unit, "count", "bool", "enum", or "" (dimensionless).
        kind: NUMERIC (float series) or CATEGORICAL (label series).
        extract: pure read-only function mapping a SampleContext to this step's value.
    """

    name: str
    group: str
    title: str
    unit: str
    kind: SignalKind
    extract: ExtractorFn


# ---------------------------------------------------------------------------
# Extractor helpers (typed factories; no getattr-style dynamic dispatch)
# ---------------------------------------------------------------------------


def _typed[M](ctx: SampleContext, message_type: type[M]) -> tuple[M, ...]:
    """Return this step's drained messages of message_type, narrowed to that type."""
    return tuple(m for m in ctx.messages.get(message_type, ()) if isinstance(m, message_type))


def _num(name: str, group: str, title: str, unit: str, extract: ExtractorFn) -> Signal:
    """Build a NUMERIC signal."""
    return Signal(name, group, title, unit, SignalKind.NUMERIC, extract)


def _cat(name: str, group: str, title: str, extract: ExtractorFn) -> Signal:
    """Build a CATEGORICAL signal (unit fixed to 'enum')."""
    return Signal(name, group, title, "enum", SignalKind.CATEGORICAL, extract)


def _count_of(message_type: type) -> ExtractorFn:
    """Extractor: how many messages of message_type were published this step."""
    return lambda ctx: float(len(ctx.messages.get(message_type, ())))


def _depth_of(message_type: type) -> ExtractorFn:
    """Extractor: current consumer backlog (queue depth) for message_type."""
    return lambda ctx: float(ctx.system.bus.queue_depth(message_type))


def _dropped_of(message_type: type) -> ExtractorFn:
    """Extractor: cumulative DROP_OLDEST drops for message_type."""
    return lambda ctx: float(ctx.system.bus.dropped_count(message_type))


def _overflow_of(message_type: type) -> ExtractorFn:
    """Extractor: cumulative NEVER_DROP soft-bound overflows for message_type."""
    return lambda ctx: float(ctx.system.bus.overflow_count(message_type))


def _fault_code_count(code: FaultCode) -> ExtractorFn:
    """Extractor: how many FaultEventMsg with this fault_code were published this step."""
    return lambda ctx: float(sum(1 for m in _typed(ctx, FaultEventMsg) if m.fault_code is code))


def _heartbeat_count(subsystem: str) -> ExtractorFn:
    """Extractor: how many HeartbeatMsg from this subsystem were published this step."""
    return lambda ctx: float(sum(1 for m in _typed(ctx, HeartbeatMsg) if m.subsystem == subsystem))


def _miss_count(subsystem: str) -> ExtractorFn:
    """Extractor: the FDIR watchdog consecutive-miss count for this subsystem."""

    def extract(ctx: SampleContext) -> SignalValue:
        entry = ctx.fault_entries.get(subsystem)
        return float(entry.miss_count) if entry is not None else _NAN

    return extract


def _pending_priority_count(priority: DownlinkPriority) -> ExtractorFn:
    """Extractor: how many queued downlink items currently carry this priority."""
    return lambda ctx: float(
        sum(1 for item in ctx.system.apps.downlink.state.pending if item.priority is priority)
    )


def _last_num[M](message_type: type[M], field: Callable[[M], float]) -> ExtractorFn:
    """Extractor: a numeric field of the last message of message_type this step (NaN if none)."""

    def extract(ctx: SampleContext) -> SignalValue:
        drained = _typed(ctx, message_type)
        return float(field(drained[-1])) if drained else _NAN

    return extract


def _last_cat[M](message_type: type[M], field: Callable[[M], str]) -> ExtractorFn:
    """Extractor: a label field of the last message of message_type this step ("" if none)."""

    def extract(ctx: SampleContext) -> SignalValue:
        drained = _typed(ctx, message_type)
        return field(drained[-1]) if drained else ""

    return extract


def _telemetry_value(subsystem: str, event_name: str, key: str) -> ExtractorFn:
    """Extractor: a numeric payload value from this step's last matching TelemetryEventMsg."""

    def extract(ctx: SampleContext) -> SignalValue:
        match = [
            m
            for m in _typed(ctx, TelemetryEventMsg)
            if m.subsystem == subsystem and m.event_name == event_name
        ]
        if not match:
            return _NAN
        value = match[-1].payload.get(key)
        return float(value) if isinstance(value, (int, float)) else _NAN

    return extract


def _bool(predicate: Callable[[SampleContext], bool]) -> ExtractorFn:
    """Extractor: 1.0/0.0 from a boolean predicate over the context."""
    return lambda ctx: 1.0 if predicate(ctx) else 0.0


# ---------------------------------------------------------------------------
# Per-group signal builders
# ---------------------------------------------------------------------------


def _system_signals() -> list[Signal]:
    """Synthetic whole-system rollup signals (mode, latch, gross message/fault counts)."""
    return [
        _num("system.step", "system", "Step index", "count", lambda ctx: float(ctx.step)),
        _num("system.t", "system", "Monotonic time", "s", lambda ctx: float(ctx.t)),
        _cat(
            "system.mode",
            "system",
            "System mode (last SafetyStateMsg)",
            _last_cat(SafetyStateMsg, lambda m: m.mode.value),
        ),
        _num(
            "system.safe_latched",
            "system",
            "SAFE latched",
            "bool",
            _bool(lambda ctx: ctx.system.apps.fault.safety.safe_latched),
        ),
        _num(
            "system.total_messages",
            "system",
            "Total bus messages this step",
            "count",
            lambda ctx: float(sum(len(v) for v in ctx.messages.values())),
        ),
        _num(
            "system.total_faults",
            "system",
            "Fault events this step",
            "count",
            _count_of(FaultEventMsg),
        ),
        _num(
            "system.total_commands",
            "system",
            "Commands this step",
            "count",
            _count_of(CommandMsg),
        ),
        _num(
            "system.total_acks",
            "system",
            "Command acks this step",
            "count",
            _count_of(CommandAckMsg),
        ),
    ]


def _bus_signals() -> list[Signal]:
    """Per-message-type publish count, queue depth, drops, overflow, plus bus totals."""
    signals: list[Signal] = []
    for message_type in MESSAGE_TYPES:
        short = message_type.__name__.removesuffix("Msg")
        signals.append(
            _num(
                f"bus.published.{short}",
                "bus",
                f"{short} published/step",
                "count",
                _count_of(message_type),
            )
        )
        signals.append(
            _num(
                f"bus.depth.{short}",
                "bus",
                f"{short} queue depth",
                "count",
                _depth_of(message_type),
            )
        )
        signals.append(
            _num(
                f"bus.dropped.{short}",
                "bus",
                f"{short} dropped (cumulative)",
                "count",
                _dropped_of(message_type),
            )
        )
        signals.append(
            _num(
                f"bus.overflow.{short}",
                "bus",
                f"{short} overflow (cumulative)",
                "count",
                _overflow_of(message_type),
            )
        )
    signals.append(
        _num(
            "bus.published.total",
            "bus",
            "All messages published/step",
            "count",
            lambda ctx: float(sum(len(v) for v in ctx.messages.values())),
        )
    )
    signals.append(
        _num(
            "bus.depth.total",
            "bus",
            "Total queue depth (all types)",
            "count",
            lambda ctx: float(sum(ctx.system.bus.queue_depth(mt) for mt in MESSAGE_TYPES)),
        )
    )
    signals.append(
        _num(
            "bus.dropped.total",
            "bus",
            "Total dropped (cumulative)",
            "count",
            lambda ctx: float(ctx.system.bus.total_dropped()),
        )
    )
    signals.append(
        _num(
            "bus.overflow.total",
            "bus",
            "Total overflow (cumulative)",
            "count",
            lambda ctx: float(ctx.system.bus.total_overflow()),
        )
    )
    signals.append(
        _num(
            "bus.types_active",
            "bus",
            "Distinct message types this step",
            "count",
            lambda ctx: float(sum(1 for v in ctx.messages.values() if v)),
        )
    )
    return signals


def _payload_signals() -> list[Signal]:
    """Payload control-state, tracking estimators, gimbal driver reads, and payload bus output."""
    return [
        _cat(
            "payload.gimbal_state",
            "payload",
            "Gimbal arbiter FSM state",
            lambda ctx: ctx.payload_state.arbiter.gimbal_state.value,
        ),
        _num(
            "payload.idle_duration_s",
            "payload",
            "Arbiter idle duration",
            "s",
            lambda ctx: float(ctx.payload_state.arbiter.idle_duration_s),
        ),
        _num(
            "payload.last_command_time",
            "payload",
            "Last arbiter command time",
            "s",
            lambda ctx: float(ctx.payload_state.arbiter.last_command_time),
        ),
        _num(
            "payload.current_target_id",
            "payload",
            "Tracked blob id",
            "id",
            lambda ctx: (
                float(ctx.payload_state.arbiter.current_target_id)
                if ctx.payload_state.arbiter.current_target_id is not None
                else _NAN
            ),
        ),
        _num(
            "payload.tracked_blobs",
            "payload",
            "Tracked blob count",
            "count",
            lambda ctx: float(len(ctx.payload_state.arbiter.tracked_blobs)),
        ),
        _num(
            "payload.scan_pan_deg",
            "payload",
            "SCAN raster pan",
            "deg",
            lambda ctx: float(ctx.payload_state.arbiter.scan_pan_deg),
        ),
        _num(
            "payload.scan_direction",
            "payload",
            "SCAN raster direction",
            "sign",
            lambda ctx: float(ctx.payload_state.arbiter.scan_direction),
        ),
        _num(
            "payload.miss_count",
            "payload",
            "TRACKING miss hysteresis",
            "count",
            lambda ctx: float(ctx.payload_state.arbiter.miss_count),
        ),
        _num(
            "payload.deadband_strikes",
            "payload",
            "Deadband strikes",
            "count",
            lambda ctx: float(ctx.payload_state.deadband_strikes),
        ),
        _num(
            "payload.commanded_az_rate_deg_s",
            "payload",
            "Commanded azimuth rate",
            "deg/s",
            lambda ctx: float(ctx.payload_state.commanded_az_rate_deg_per_s),
        ),
        _num(
            "payload.commanded_el_rate_deg_s",
            "payload",
            "Commanded elevation rate",
            "deg/s",
            lambda ctx: float(ctx.payload_state.commanded_el_rate_deg_per_s),
        ),
        _num(
            "payload.ema_centroid_x",
            "payload",
            "EMA boresight error x",
            "deg",
            lambda ctx: float(ctx.payload_state.ema.centroid[0]),
        ),
        _num(
            "payload.ema_centroid_y",
            "payload",
            "EMA boresight error y",
            "deg",
            lambda ctx: float(ctx.payload_state.ema.centroid[1]),
        ),
        _num(
            "payload.ema_initialized",
            "payload",
            "EMA initialized",
            "bool",
            _bool(lambda ctx: ctx.payload_state.ema.initialized),
        ),
        _num(
            "payload.kalman_az_err",
            "payload",
            "Kalman azimuth error",
            "deg",
            lambda ctx: float(ctx.payload_state.kalman.x[0]),
        ),
        _num(
            "payload.kalman_el_err",
            "payload",
            "Kalman elevation error",
            "deg",
            lambda ctx: float(ctx.payload_state.kalman.x[1]),
        ),
        _num(
            "payload.kalman_az_vel",
            "payload",
            "Kalman azimuth rate",
            "deg/s",
            lambda ctx: float(ctx.payload_state.kalman.x[2]),
        ),
        _num(
            "payload.kalman_el_vel",
            "payload",
            "Kalman elevation rate",
            "deg/s",
            lambda ctx: float(ctx.payload_state.kalman.x[3]),
        ),
        _num(
            "payload.kalman_p_trace",
            "payload",
            "Kalman covariance trace",
            "deg^2",
            lambda ctx: float(np.trace(ctx.payload_state.kalman.P)),
        ),
        _num(
            "payload.runaway_strikes",
            "payload",
            "Encoder runaway strikes",
            "count",
            lambda ctx: float(ctx.payload_state.runaway.strike_count),
        ),
        _num(
            "payload.motion_inhibited",
            "payload",
            "Launch-lock motion inhibit",
            "bool",
            _bool(lambda ctx: ctx.system.apps.payload.lock_gate.engaged),
        ),
        _num(
            "payload.gimbal_az_meas_deg",
            "payload",
            "Gimbal azimuth (measured)",
            "deg",
            lambda ctx: ctx.devices.gimbal_az_meas_deg,
        ),
        _num(
            "payload.gimbal_el_meas_deg",
            "payload",
            "Gimbal elevation (measured)",
            "deg",
            lambda ctx: ctx.devices.gimbal_el_meas_deg,
        ),
        _num(
            "payload.gimbal_az_true_deg",
            "payload",
            "Gimbal azimuth (truth)",
            "deg",
            lambda ctx: ctx.devices.gimbal_az_true_deg,
        ),
        _num(
            "payload.gimbal_el_true_deg",
            "payload",
            "Gimbal elevation (truth)",
            "deg",
            lambda ctx: ctx.devices.gimbal_el_true_deg,
        ),
        _num(
            "payload.gimbal_rate_az_deg_s",
            "payload",
            "Gimbal azimuth rate (driver)",
            "deg/s",
            lambda ctx: ctx.devices.gimbal_rate_az_deg_s,
        ),
        _num(
            "payload.gimbal_rate_el_deg_s",
            "payload",
            "Gimbal elevation rate (driver)",
            "deg/s",
            lambda ctx: ctx.devices.gimbal_rate_el_deg_s,
        ),
        _cat(
            "payload.gimbal_driver_mode",
            "payload",
            "Gimbal driver command mode",
            lambda ctx: ctx.devices.gimbal_mode,
        ),
        _num(
            "payload.stow_switch",
            "payload",
            "Stow switch engaged",
            "bool",
            _bool(lambda ctx: ctx.devices.stow_switch),
        ),
        _num(
            "payload.sensor_index",
            "payload",
            "Frames acquired (replay cursor)",
            "count",
            lambda ctx: float(ctx.devices.sensor_index),
        ),
        _num(
            "payload.inference_count",
            "payload",
            "Inference results/step",
            "count",
            _count_of(InferenceResultMsg),
        ),
        _num(
            "payload.inference_blobs",
            "payload",
            "Blobs in last inference",
            "count",
            _last_num(InferenceResultMsg, lambda m: float(len(m.blobs))),
        ),
        _num(
            "payload.inference_ms",
            "payload",
            "Last inference duration",
            "ms",
            _last_num(InferenceResultMsg, lambda m: m.inference_ms),
        ),
        _num(
            "payload.gimbal_command_count",
            "payload",
            "Gimbal commands/step",
            "count",
            _count_of(GimbalCommandMsg),
        ),
        _cat(
            "payload.gimbal_command_mode",
            "payload",
            "Last gimbal command mode",
            _last_cat(GimbalCommandMsg, lambda m: m.mode.value),
        ),
        _num(
            "payload.gimbal_command_az",
            "payload",
            "Last gimbal command az value",
            "deg",
            _last_num(GimbalCommandMsg, lambda m: m.az_value_deg),
        ),
        _num(
            "payload.gimbal_command_el",
            "payload",
            "Last gimbal command el value",
            "deg",
            _last_num(GimbalCommandMsg, lambda m: m.el_value_deg),
        ),
        _num(
            "payload.product_ref_count",
            "payload",
            "Science products/step",
            "count",
            _count_of(ProductRefMsg),
        ),
        _num(
            "payload.fault_count",
            "payload",
            "Payload faults/step",
            "count",
            lambda ctx: float(
                sum(1 for m in _typed(ctx, FaultEventMsg) if m.subsystem == "payload")
            ),
        ),
        _num(
            "payload.telemetry_count",
            "payload",
            "Payload telemetry/step",
            "count",
            lambda ctx: float(
                sum(1 for m in _typed(ctx, TelemetryEventMsg) if m.subsystem == "payload")
            ),
        ),
    ]


def _fault_signals() -> list[Signal]:
    """FDIR latch + safety state, mode changes, per-subsystem watchdog, and per-code faults."""
    signals: list[Signal] = [
        _num(
            "fault.safe_latched",
            "fault",
            "SAFE latched",
            "bool",
            _bool(lambda ctx: ctx.system.apps.fault.safety.safe_latched),
        ),
        _cat(
            "fault.safe_reason",
            "fault",
            "Latched SAFE reason",
            lambda ctx: ctx.system.apps.fault.safety.safe_reason.value,
        ),
        _cat(
            "fault.safety_mode",
            "fault",
            "SafetyStateMsg mode (last)",
            _last_cat(SafetyStateMsg, lambda m: m.mode.value),
        ),
        _num(
            "fault.safety_active_faults",
            "fault",
            "Active SAFE-triggering faults (last)",
            "count",
            _last_num(SafetyStateMsg, lambda m: float(len(m.active_faults))),
        ),
        _num(
            "fault.safety_msg_count",
            "fault",
            "SafetyStateMsg/step",
            "count",
            _count_of(SafetyStateMsg),
        ),
        _num(
            "fault.mode_change_count",
            "fault",
            "ModeChangeMsg/step",
            "count",
            _count_of(ModeChangeMsg),
        ),
        _cat(
            "fault.last_mode_change",
            "fault",
            "Last requested mode",
            _last_cat(ModeChangeMsg, lambda m: m.new_mode.value),
        ),
        _num("fault.event_count", "fault", "Fault events/step", "count", _count_of(FaultEventMsg)),
    ]
    for subsystem in MONITORED:
        signals.append(
            _num(
                f"fault.miss.{subsystem}",
                "fault",
                f"{subsystem} watchdog misses",
                "count",
                _miss_count(subsystem),
            )
        )
    for subsystem in MONITORED:
        signals.append(
            _num(
                f"fault.heartbeats.{subsystem}",
                "fault",
                f"{subsystem} heartbeats/step",
                "count",
                _heartbeat_count(subsystem),
            )
        )
    for code in FaultCode:
        signals.append(
            _num(
                f"fault.code.{code.value}",
                "fault",
                f"{code.value} faults/step",
                "count",
                _fault_code_count(code),
            )
        )
    return signals


def _iss_iface_signals() -> list[Signal]:
    """ISS interface: TM sequence, ingress replay guard, upload buffer, link, station traffic."""
    return [
        _num(
            "iss_iface.tm_sequence",
            "iss_iface",
            "Outbound TM sequence",
            "count",
            lambda ctx: float(ctx.system.apps.iss_iface.state.tm_sequence),
        ),
        _num(
            "iss_iface.known_sources",
            "iss_iface",
            "Replay-guard known sources",
            "count",
            lambda ctx: float(len(ctx.system.apps.iss_iface.state.last_seq)),
        ),
        _num(
            "iss_iface.upload_chunks_buffered",
            "iss_iface",
            "Upload chunks buffered",
            "count",
            lambda ctx: float(len(ctx.system.apps.iss_iface.state.upload.chunks)),
        ),
        _num(
            "iss_iface.upload_total_chunks",
            "iss_iface",
            "Upload total chunks",
            "count",
            lambda ctx: float(ctx.system.apps.iss_iface.state.upload.total_chunks),
        ),
        _cat(
            "iss_iface.link_state",
            "iss_iface",
            "Station link state",
            lambda ctx: ctx.devices.link_state,
        ),
        _num(
            "iss_iface.station_sent_total",
            "iss_iface",
            "Station packets sent (cumulative)",
            "count",
            lambda ctx: float(ctx.devices.station_sent_total),
        ),
        _num(
            "iss_iface.link_state_count",
            "iss_iface",
            "LinkStateMsg/step",
            "count",
            _count_of(LinkStateMsg),
        ),
        _num(
            "iss_iface.command_published",
            "iss_iface",
            "Validated commands published/step",
            "count",
            _count_of(CommandMsg),
        ),
        _num(
            "iss_iface.ack_count",
            "iss_iface",
            "Command acks/step",
            "count",
            _count_of(CommandAckMsg),
        ),
        _num(
            "iss_iface.model_staged_count",
            "iss_iface",
            "ModelStagedMsg/step",
            "count",
            _count_of(ModelStagedMsg),
        ),
        _num(
            "iss_iface.upload_chunk_count",
            "iss_iface",
            "Upload chunks/step",
            "count",
            _count_of(UploadChunkMsg),
        ),
    ]


def _thermal_signals() -> list[Signal]:
    """Thermal: temperature, effective + override limit, over-limit flag, sample/fault counts."""
    return [
        _num(
            "thermal.temperature_c",
            "thermal",
            "Temperature",
            "degC",
            _telemetry_value("thermal", "thermal_sample", "temperature_c"),
        ),
        _num(
            "thermal.limit_c",
            "thermal",
            "Effective thermal limit",
            "degC",
            lambda ctx: float(
                ctx.system.apps.thermal.state.limit_c_override
                if ctx.system.apps.thermal.state.limit_c_override is not None
                else ctx.system.apps.thermal.cfg.thermal_limit_c
            ),
        ),
        _num(
            "thermal.limit_override_c",
            "thermal",
            "Commanded limit override",
            "degC",
            lambda ctx: (
                float(ctx.system.apps.thermal.state.limit_c_override)
                if ctx.system.apps.thermal.state.limit_c_override is not None
                else _NAN
            ),
        ),
        _num(
            "thermal.sample_count",
            "thermal",
            "Thermal samples/step",
            "count",
            lambda ctx: float(
                sum(
                    1
                    for m in _typed(ctx, TelemetryEventMsg)
                    if m.subsystem == "thermal" and m.event_name == "thermal_sample"
                )
            ),
        ),
        _num(
            "thermal.fault_count",
            "thermal",
            "Thermal over-limit faults/step",
            "count",
            lambda ctx: float(
                sum(
                    1
                    for m in _typed(ctx, FaultEventMsg)
                    if m.fault_code is FaultCode.THERMAL_OVER_LIMIT
                )
            ),
        ),
        _num(
            "thermal.power_index",
            "thermal",
            "Thermal replay cursor",
            "count",
            lambda ctx: float(ctx.devices.thermal_index),
        ),
    ]


def _electrical_signals() -> list[Signal]:
    """Electrical: bus power, power limit, over-limit flag, sample/fault counts."""
    return [
        _num(
            "electrical.power_w",
            "electrical",
            "Bus power",
            "W",
            _telemetry_value("electrical", "electrical_sample", "power_w"),
        ),
        _num(
            "electrical.limit_w",
            "electrical",
            "Power limit",
            "W",
            lambda ctx: float(ctx.system.apps.electrical.cfg.power_limit_w),
        ),
        _num(
            "electrical.sample_count",
            "electrical",
            "Electrical samples/step",
            "count",
            lambda ctx: float(
                sum(
                    1
                    for m in _typed(ctx, TelemetryEventMsg)
                    if m.subsystem == "electrical" and m.event_name == "electrical_sample"
                )
            ),
        ),
        _num(
            "electrical.fault_count",
            "electrical",
            "Power over-limit faults/step",
            "count",
            lambda ctx: float(
                sum(
                    1
                    for m in _typed(ctx, FaultEventMsg)
                    if m.fault_code is FaultCode.POWER_OVER_LIMIT
                )
            ),
        ),
        _num(
            "electrical.power_index",
            "electrical",
            "Power replay cursor",
            "count",
            lambda ctx: float(ctx.devices.power_index),
        ),
    ]


def _command_router_signals() -> list[Signal]:
    """Command router: armed hazardous commands, SAFE mirror, routing throughput."""
    return [
        _num(
            "command_router.armed",
            "command_router",
            "Armed hazardous commands",
            "count",
            lambda ctx: float(len(ctx.system.apps.command_router.state.armed)),
        ),
        _num(
            "command_router.safe_latched",
            "command_router",
            "SAFE mirror",
            "bool",
            _bool(lambda ctx: ctx.system.apps.command_router.state.safe_latched),
        ),
        _num(
            "command_router.routed_count",
            "command_router",
            "Routed commands/step",
            "count",
            _count_of(RoutedCommandMsg),
        ),
        _num(
            "command_router.command_count",
            "command_router",
            "Commands seen/step",
            "count",
            _count_of(CommandMsg),
        ),
        _num(
            "command_router.unroutable_count",
            "command_router",
            "Unroutable faults/step",
            "count",
            lambda ctx: float(
                sum(
                    1
                    for m in _typed(ctx, FaultEventMsg)
                    if m.fault_code is FaultCode.COMMAND_UNROUTABLE
                )
            ),
        ),
        _num(
            "command_router.ack_count",
            "command_router",
            "Command acks/step",
            "count",
            _count_of(CommandAckMsg),
        ),
    ]


def _storage_signals() -> list[Signal]:
    """Storage: live bytes/entries, monotonic insert + eviction counters, fullness, write flow."""
    return [
        _num(
            "storage.total_bytes",
            "storage",
            "Live stored bytes",
            "bytes",
            lambda ctx: float(ctx.system.apps.storage.state.total_bytes),
        ),
        _num(
            "storage.entries",
            "storage",
            "Live stored entries",
            "count",
            lambda ctx: float(len(ctx.system.apps.storage.state.entries)),
        ),
        _num(
            "storage.next_order",
            "storage",
            "Entries ever stored (cumulative)",
            "count",
            lambda ctx: float(ctx.system.apps.storage.state.next_order),
        ),
        _num(
            "storage.dropped_count",
            "storage",
            "Entries evicted (cumulative)",
            "count",
            lambda ctx: float(ctx.system.apps.storage.state.dropped_count),
        ),
        _num(
            "storage.fraction_full",
            "storage",
            "Quota fraction used",
            "fraction",
            lambda ctx: (
                float(ctx.system.apps.storage.state.total_bytes)
                / float(ctx.system.apps.storage.cfg.max_storage_bytes)
            ),
        ),
        _num(
            "storage.write_count",
            "storage",
            "Storage writes/step",
            "count",
            _count_of(StorageWriteMsg),
        ),
        _num(
            "storage.full_fault_count",
            "storage",
            "Storage-full faults/step",
            "count",
            lambda ctx: float(
                sum(1 for m in _typed(ctx, FaultEventMsg) if m.fault_code is FaultCode.STORAGE_FULL)
            ),
        ),
        _num(
            "storage.telemetry_persisted",
            "storage",
            "Telemetry persisted/step",
            "count",
            _count_of(TelemetryEventMsg),
        ),
    ]


def _downlink_signals() -> list[Signal]:
    """Downlink: queue depth/bytes, AOS gate, per-priority backlog, emission throughput, budget."""
    signals = [
        _num(
            "downlink.pending_items",
            "downlink",
            "Queued downlink items",
            "count",
            lambda ctx: float(len(ctx.system.apps.downlink.state.pending)),
        ),
        _num(
            "downlink.pending_bytes",
            "downlink",
            "Queued downlink bytes",
            "bytes",
            lambda ctx: float(
                sum(item.byte_len for item in ctx.system.apps.downlink.state.pending)
            ),
        ),
        _num(
            "downlink.next_order",
            "downlink",
            "Items ever enqueued (cumulative)",
            "count",
            lambda ctx: float(ctx.system.apps.downlink.state.next_order),
        ),
        _num(
            "downlink.aos",
            "downlink",
            "AOS (link up)",
            "bool",
            _bool(lambda ctx: ctx.system.apps.downlink.state.aos),
        ),
        _num(
            "downlink.item_count",
            "downlink",
            "DownlinkItemMsg/step",
            "count",
            _count_of(DownlinkItemMsg),
        ),
        _num(
            "downlink.budget_bytes_per_pass",
            "downlink",
            "Per-pass byte budget",
            "bytes",
            lambda ctx: float(ctx.system.apps.downlink.cfg.downlink_max_bytes_per_pass),
        ),
    ]
    for priority in DownlinkPriority:
        signals.append(
            _num(
                f"downlink.pending.{priority.name}",
                "downlink",
                f"Queued {priority.name} items",
                "count",
                _pending_priority_count(priority),
            )
        )
    return signals


def _mechanical_signals() -> list[Signal]:
    """Mechanical: launch-lock state + interlock telemetry, observed gimbal motion."""
    return [
        _cat(
            "mechanical.launch_lock_state",
            "mechanical",
            "Launch-lock state",
            lambda ctx: ctx.system.apps.mechanical.state.last_state.value,
        ),
        _num(
            "mechanical.launch_lock_engaged",
            "mechanical",
            "Launch-lock engaged",
            "bool",
            _bool(lambda ctx: ctx.devices.launch_lock_state == "ENGAGED"),
        ),
        _num(
            "mechanical.lock_state_msg_count",
            "mechanical",
            "LaunchLockStateMsg/step",
            "count",
            _count_of(LaunchLockStateMsg),
        ),
        _num(
            "mechanical.lock_fault_count",
            "mechanical",
            "Launch-lock faults/step",
            "count",
            lambda ctx: float(
                sum(
                    1
                    for m in _typed(ctx, FaultEventMsg)
                    if m.fault_code is FaultCode.LAUNCH_LOCK_FAULT
                )
            ),
        ),
        _num(
            "mechanical.gimbal_cmd_observed",
            "mechanical",
            "Gimbal commands observed/step",
            "count",
            _count_of(GimbalCommandMsg),
        ),
    ]


def _model_deploy_signals() -> list[Signal]:
    """Model deploy: lifecycle state, active/rollback versions, staged artifact, transitions."""
    return [
        _cat(
            "model_deploy.state",
            "model_deploy",
            "Deploy lifecycle state",
            lambda ctx: ctx.system.apps.model_deploy.state.state.value,
        ),
        _cat(
            "model_deploy.active_version",
            "model_deploy",
            "Active model version",
            lambda ctx: ctx.system.apps.model_deploy.state.active_version,
        ),
        _num(
            "model_deploy.has_rollback",
            "model_deploy",
            "Rollback model retained",
            "bool",
            _bool(lambda ctx: ctx.system.apps.model_deploy.state.rollback_version is not None),
        ),
        _num(
            "model_deploy.has_staged",
            "model_deploy",
            "Staged model awaiting activate",
            "bool",
            _bool(lambda ctx: ctx.system.apps.model_deploy.state.staged is not None),
        ),
        _num(
            "model_deploy.staged_input_dims",
            "model_deploy",
            "Staged input shape rank",
            "count",
            lambda ctx: (
                float(len(ctx.system.apps.model_deploy.state.staged.input_shape))
                if ctx.system.apps.model_deploy.state.staged is not None
                else _NAN
            ),
        ),
        _num(
            "model_deploy.state_msg_count",
            "model_deploy",
            "ModelDeployStateMsg/step",
            "count",
            _count_of(ModelDeployStateMsg),
        ),
        _num(
            "model_deploy.corrupt_fault_count",
            "model_deploy",
            "Model-corrupt faults/step",
            "count",
            lambda ctx: float(
                sum(
                    1 for m in _typed(ctx, FaultEventMsg) if m.fault_code is FaultCode.MODEL_CORRUPT
                )
            ),
        ),
    ]


def _enrichment_signals() -> list[Signal]:
    """Extra derived/observability signals enriching the per-group coverage.

    These read the same context as the per-group builders but add second-order quantities worth
    a series of their own: the Kalman covariance diagonal, per-subsystem heartbeat age and the
    watchdog interval budget (so the watchdog/process-died path is fully observable), and a few
    derived fractions/magnitudes. Each carries its owning group so it lands in that group's wide
    frame and figures.
    """
    signals: list[Signal] = [
        _num(
            "payload.kalman_p00",
            "payload",
            "Kalman P[0,0] (az err var)",
            "deg^2",
            lambda ctx: float(ctx.payload_state.kalman.P[0, 0]),
        ),
        _num(
            "payload.kalman_p11",
            "payload",
            "Kalman P[1,1] (el err var)",
            "deg^2",
            lambda ctx: float(ctx.payload_state.kalman.P[1, 1]),
        ),
        _num(
            "payload.kalman_p22",
            "payload",
            "Kalman P[2,2] (az rate var)",
            "deg^2/s^2",
            lambda ctx: float(ctx.payload_state.kalman.P[2, 2]),
        ),
        _num(
            "payload.kalman_p33",
            "payload",
            "Kalman P[3,3] (el rate var)",
            "deg^2/s^2",
            lambda ctx: float(ctx.payload_state.kalman.P[3, 3]),
        ),
        _num(
            "payload.ema_error_mag",
            "payload",
            "EMA boresight error magnitude",
            "deg",
            lambda ctx: float(
                math.hypot(ctx.payload_state.ema.centroid[0], ctx.payload_state.ema.centroid[1])
            ),
        ),
        _num(
            "payload.gimbal_az_noise_deg",
            "payload",
            "Gimbal az measure-minus-truth",
            "deg",
            lambda ctx: ctx.devices.gimbal_az_meas_deg - ctx.devices.gimbal_az_true_deg,
        ),
        _num(
            "payload.is_tracking",
            "payload",
            "Arbiter in TRACKING",
            "bool",
            _bool(lambda ctx: ctx.payload_state.arbiter.gimbal_state.value == "TRACKING"),
        ),
        _num(
            "iss_iface.upload_progress",
            "iss_iface",
            "Upload reassembly fraction",
            "fraction",
            lambda ctx: (
                float(len(ctx.system.apps.iss_iface.state.upload.chunks))
                / float(ctx.system.apps.iss_iface.state.upload.total_chunks)
                if ctx.system.apps.iss_iface.state.upload.total_chunks > 0
                else 0.0
            ),
        ),
        _num(
            "storage.headroom_bytes",
            "storage",
            "Storage quota headroom",
            "bytes",
            lambda ctx: (
                float(ctx.system.apps.storage.cfg.max_storage_bytes)
                - float(ctx.system.apps.storage.state.total_bytes)
            ),
        ),
        _num(
            "storage.avg_entry_bytes",
            "storage",
            "Mean live entry size",
            "bytes",
            lambda ctx: (
                float(ctx.system.apps.storage.state.total_bytes)
                / float(len(ctx.system.apps.storage.state.entries))
                if ctx.system.apps.storage.state.entries
                else _NAN
            ),
        ),
        _num(
            "downlink.backlog_fraction",
            "downlink",
            "Backlog vs per-pass budget",
            "fraction",
            lambda ctx: (
                float(sum(i.byte_len for i in ctx.system.apps.downlink.state.pending))
                / float(ctx.system.apps.downlink.cfg.downlink_max_bytes_per_pass)
                if ctx.system.apps.downlink.cfg.downlink_max_bytes_per_pass > 0
                else _NAN
            ),
        ),
        _num(
            "model_deploy.staged_output_dims",
            "model_deploy",
            "Staged output shape rank",
            "count",
            lambda ctx: (
                float(len(ctx.system.apps.model_deploy.state.staged.output_shape))
                if ctx.system.apps.model_deploy.state.staged is not None
                else _NAN
            ),
        ),
        _num(
            "model_deploy.active_is_factory",
            "model_deploy",
            "Active model is factory",
            "bool",
            _bool(lambda ctx: ctx.system.apps.model_deploy.state.active_version == "factory"),
        ),
    ]
    for subsystem in MONITORED:
        signals.append(
            _num(
                f"fault.heartbeat_age.{subsystem}",
                "fault",
                f"{subsystem} heartbeat age",
                "s",
                _heartbeat_age(subsystem),
            )
        )
    for subsystem in MONITORED:
        signals.append(
            _num(
                f"fault.max_interval.{subsystem}",
                "fault",
                f"{subsystem} watchdog interval",
                "s",
                _max_interval(subsystem),
            )
        )
    return signals


def _heartbeat_age(subsystem: str) -> ExtractorFn:
    """Extractor: seconds since this subsystem's last watchdog-registered heartbeat."""

    def extract(ctx: SampleContext) -> SignalValue:
        entry = ctx.fault_entries.get(subsystem)
        return float(ctx.t - entry.last_heartbeat_time) if entry is not None else _NAN

    return extract


def _max_interval(subsystem: str) -> ExtractorFn:
    """Extractor: this subsystem's watchdog max-interval budget (seconds before a miss)."""

    def extract(ctx: SampleContext) -> SignalValue:
        entry = ctx.fault_entries.get(subsystem)
        return float(entry.max_interval_s) if entry is not None else _NAN

    return extract


def build_registry() -> tuple[Signal, ...]:
    """Assemble the full per-step signal registry across all groups.

    Returns:
        A tuple of every registered Signal, ordered by group then declaration. Building it via a
        function (rather than a module-level literal) keeps the per-group builders independently
        testable and the assembly order explicit.

    Raises:
        ValueError: if two signals share a name (the registry must be uniquely keyed).
    """
    signals: list[Signal] = []
    signals.extend(_system_signals())
    signals.extend(_bus_signals())
    signals.extend(_payload_signals())
    signals.extend(_fault_signals())
    signals.extend(_iss_iface_signals())
    signals.extend(_thermal_signals())
    signals.extend(_electrical_signals())
    signals.extend(_command_router_signals())
    signals.extend(_storage_signals())
    signals.extend(_downlink_signals())
    signals.extend(_mechanical_signals())
    signals.extend(_model_deploy_signals())
    signals.extend(_enrichment_signals())
    seen: set[str] = set()
    for signal in signals:
        if signal.name in seen:
            raise ValueError(f"duplicate signal name: {signal.name}")
        seen.add(signal.name)
    return tuple(signals)


REGISTRY: tuple[Signal, ...] = build_registry()

# Ordered, de-duplicated group names (every flight app + the bus + the system rollup).
GROUPS: tuple[str, ...] = tuple(dict.fromkeys(signal.group for signal in REGISTRY))


def signals_for_group(group: str) -> tuple[Signal, ...]:
    """Return every registered Signal in the given group, in registry order."""
    return tuple(signal for signal in REGISTRY if signal.group == group)


def signal_names() -> tuple[str, ...]:
    """Return every registered signal name, in registry order."""
    return tuple(signal.name for signal in REGISTRY)


def is_event_rate(signal: Signal) -> bool:
    """Return True if signal is a per-step event count (its cumulative curve is meaningful).

    Discriminated purely by the title convention ("... /step" or "... this step") so level
    signals -- queue depths, byte totals, monotonic cursors -- are never cumulatively summed.
    """
    return signal.kind is SignalKind.NUMERIC and (
        signal.title.endswith("/step") or signal.title.endswith("this step")
    )


def accumulable_names() -> tuple[str, ...]:
    """Return the names of every per-step event-count signal, in registry order.

    The recorder derives a ``<name>.cumulative`` running-total series for each of these so the
    report carries both the per-step rate and the cumulative count (a standard mission-ops view).
    """
    return tuple(signal.name for signal in REGISTRY if is_event_rate(signal))


def is_nan(value: SignalValue) -> bool:
    """Return True if value is a float NaN (a failed-extractor sentinel)."""
    return isinstance(value, float) and math.isnan(value)
