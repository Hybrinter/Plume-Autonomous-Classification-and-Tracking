"""Passive per-step capture loop: drive the deterministic SIL and tabulate every signal.

The recorder owns the SIL stepping loop (it reuses ``sim.sil.step_once`` -- the single source of
truth for one cycle -- and threads the payload ``ControlState`` + FDIR watchdog entries itself), so
it can observe the state those two apps keep off ``self`` (threaded through ``step_once``). Capture
is fully passive: it subscribes to all nineteen message types on the shared bus, and because the
bus is fan-out (each subscriber gets its own queue) draining the recorder's subscriptions steals
nothing from the apps. Each step it drains its subscriptions, takes one self-consistent
``DeviceSample`` from ``SimGimbal.snapshot`` and other sim driver fields, builds a
``SampleContext``, and evaluates every registered signal; an extractor that raises is
recorded as NaN (numeric) or "" (categorical).

The result is a tidy long frame (one row per step per emitted column) plus one wide frame per group
(step-indexed, one column per signal), with a cumulative running-total column added for each
per-step event-count signal. Nothing here mutates flight state.

Contains:
  - CaptureResult: the long + per-group wide frames and the run's column/step counts.
  - sample_devices: non-mutating snapshot of last delivered encoder and plant truth.
  - record_run: run the passive capture loop and return a CaptureResult.

Satisfies: REQ-OBS-SIL-001.
"""

from __future__ import annotations

# stdlib
from collections.abc import Callable
from dataclasses import dataclass

# third-party
import pandas as pd

# internal
from flight.libs.bus import Subscription
from sim.sil import SilSystem, step_once

from tools.analysis.datapoints import (
    GROUPS,
    MESSAGE_TYPES,
    REGISTRY,
    DeviceSample,
    SampleContext,
    Signal,
    SignalKind,
    SignalValue,
    accumulable_names,
)

PreStepHook = Callable[[SilSystem, int], None]
_CUMULATIVE_SUFFIX = ".cumulative"


@dataclass(frozen=True, slots=True)
class CaptureResult:
    """The tabulated output of one passive capture run.

    Fields:
        long: tidy long frame with columns step, t, group, signal, unit, kind, value_num,
            value_str (one row per step per emitted column).
        wide: per-group wide frames keyed by group name, step-indexed, with a leading ``t`` column
            and one column per signal (and its ``.cumulative`` derivation where applicable).
        n_steps: number of steps captured.
        n_signals: number of registry signals evaluated each step.
        n_columns: total emitted value columns (registry signals + cumulative derivations).
    """

    long: pd.DataFrame
    wide: dict[str, pd.DataFrame]
    n_steps: int
    n_signals: int
    n_columns: int


def _drain(subscription: Subscription[object]) -> tuple[object, ...]:
    """Drain every pending message from a passive subscription into a tuple."""
    drained: list[object] = []
    while not subscription.empty():
        drained.append(subscription.get_nowait())
    return tuple(drained)


def sample_devices(system: SilSystem) -> DeviceSample:
    """Take a non-mutating snapshot of sim HAL drivers for the current step.

    Args:
        system: The wired SilSystem whose sim drivers to read.

    Returns:
        A DeviceSample. ``gimbal_el_meas_deg`` is the last encoder elevation
        ``read_position`` delivered to flight (NaN if none yet), not a new draw.

    Notes:
        Gimbal fields come from ``SimGimbal.snapshot``. Observation does not
        integrate the plant, draw encoder noise, update last-feedback time,
        expire a command lease, or call ``read_stow_switch``.
    """
    gimbal = system.gimbal
    snap = gimbal.snapshot()
    pose_mode = "STOW" if snap.stow_commanded else "NONE"
    return DeviceSample(
        gimbal_el_meas_deg=snap.last_el_meas_deg,
        gimbal_el_true_deg=snap.true_el_deg,
        gimbal_omega_rad_s=snap.true_omega_rad_s,
        gimbal_tau_nm=snap.tau_nm,
        gimbal_mode=pose_mode,
        stow_switch=snap.stow_switch,
        link_state=system.station.link_state().value,
        station_sent_total=len(system.station.sent),
        sensor_index=system.sensor._index,
        thermal_index=system.thermal_sensor._index,
        power_index=system.power_sensor._index,
    )


def _evaluate(signal: Signal, ctx: SampleContext) -> SignalValue:
    """Evaluate one signal against a context, mapping any extractor failure to a NaN/"" sentinel."""
    try:
        value = signal.extract(ctx)
    except Exception:
        return float("nan") if signal.kind is SignalKind.NUMERIC else ""
    if signal.kind is SignalKind.NUMERIC:
        if isinstance(value, (int, float)):
            return float(value)
        return float("nan")
    return value if isinstance(value, str) else ""


def record_run(
    system: SilSystem,
    steps: int,
    dt: float = 1.0,
    pre_step: PreStepHook | None = None,
) -> CaptureResult:
    """Run the passive capture loop over a wired SilSystem and tabulate every signal each step.

    Args:
        system: The wired SilSystem to drive (apps + bus + sim drivers + clock).
        steps: Number of deterministic steps to run (must be positive).
        dt: Seconds to advance the shared clock and ``now`` per step.
        pre_step: Optional hook called as ``pre_step(system, step)`` just before each step's
            ``step_once`` -- the scenario runner uses it to publish timed bus injections or run
            timed actions. It must not consume the recorder's subscriptions.

    Returns:
        A CaptureResult with the tidy long frame and the per-group wide frames.

    Raises:
        ValueError: if steps is not positive.

    Notes:
        Subscriptions are created before the first step, so step 1 is captured. The loop mirrors
        SilHarness.run_steps (``now``, optional hook, ``step_once``, then clock advance) and owns
        the threaded state so the payload/FDIR internals are observable. ``sample_devices`` uses
        ``SimGimbal.snapshot`` and does not call ``read_position`` or ``read_stow_switch``.
    """
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    subscriptions: dict[type, Subscription[object]] = {
        message_type: system.bus.subscribe(message_type) for message_type in MESSAGE_TYPES
    }
    payload_state = system.apps.payload.controller.initial_state()
    fault_entries = system.apps.fault.initial_entries()

    step_index: list[int] = []
    t_values: list[float] = []
    columns: dict[str, list[SignalValue]] = {signal.name: [] for signal in REGISTRY}

    now = 0.0
    for step in range(1, steps + 1):
        now += dt
        if pre_step is not None:
            pre_step(system, step)
        payload_state, fault_entries = step_once(
            system.apps,
            system.sensor,
            system.gimbal,
            system.bus,
            system.clock,
            now,
            payload_state,
            fault_entries,
        )
        system.clock.advance(dt)
        messages: dict[type, tuple[object, ...]] = {
            message_type: _drain(subscription)
            for message_type, subscription in subscriptions.items()
        }
        devices = sample_devices(system)
        ctx = SampleContext(step, now, system, payload_state, fault_entries, messages, devices)
        step_index.append(step)
        t_values.append(now)
        for signal in REGISTRY:
            columns[signal.name].append(_evaluate(signal, ctx))

    return _build_result(step_index, t_values, columns)


def _build_result(
    step_index: list[int], t_values: list[float], columns: dict[str, list[SignalValue]]
) -> CaptureResult:
    """Assemble the master frame, cumulative derivations, per-group wide frames, and long frame."""
    index = pd.Index(step_index, name="step")
    base: dict[str, list[SignalValue] | list[float]] = {"t": list(t_values)}
    for signal in REGISTRY:
        base[signal.name] = columns[signal.name]
    master = pd.DataFrame(base, index=index)

    column_group: dict[str, str] = {signal.name: signal.group for signal in REGISTRY}
    column_unit: dict[str, str] = {signal.name: signal.unit for signal in REGISTRY}
    column_kind: dict[str, SignalKind] = {signal.name: signal.kind for signal in REGISTRY}
    cumulative_frames: dict[str, pd.Series] = {}
    for name in accumulable_names():
        cumulative = f"{name}{_CUMULATIVE_SUFFIX}"
        cumulative_frames[cumulative] = master[name].cumsum()
        column_group[cumulative] = column_group[name]
        column_unit[cumulative] = "count"
        column_kind[cumulative] = SignalKind.NUMERIC
    if cumulative_frames:
        master = pd.concat([master, pd.DataFrame(cumulative_frames, index=index)], axis=1)

    value_columns = [c for c in master.columns if c != "t"]
    wide: dict[str, pd.DataFrame] = {}
    for group in GROUPS:
        group_columns = [c for c in value_columns if column_group[c] == group]
        wide[group] = master[["t", *group_columns]].copy()

    long = _build_long(master, value_columns, column_group, column_unit, column_kind)
    return CaptureResult(
        long=long,
        wide=wide,
        n_steps=len(step_index),
        n_signals=len(REGISTRY),
        n_columns=len(value_columns),
    )


def _build_long(
    master: pd.DataFrame,
    value_columns: list[str],
    column_group: dict[str, str],
    column_unit: dict[str, str],
    column_kind: dict[str, SignalKind],
) -> pd.DataFrame:
    """Reshape the wide master frame into one tidy long frame (numeric + categorical kept apart)."""
    steps = master.index.to_numpy()
    t_array = master["t"].to_numpy()
    parts: list[pd.DataFrame] = []
    for column in value_columns:
        kind = column_kind[column]
        series = master[column]
        part = pd.DataFrame(
            {
                "step": steps,
                "t": t_array,
                "group": column_group[column],
                "signal": column,
                "unit": column_unit[column],
                "kind": kind.value,
            }
        )
        if kind is SignalKind.NUMERIC:
            part["value_num"] = series.to_numpy(dtype="float64")
            part["value_str"] = ""
        else:
            part["value_num"] = float("nan")
            part["value_str"] = series.astype("string").to_numpy()
        parts.append(part)
    return pd.concat(parts, ignore_index=True)
