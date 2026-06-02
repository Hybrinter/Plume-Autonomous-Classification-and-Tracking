# Phase 5e: Payload Application Shell Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bind the HAL drivers and the pure payload core into one in-process payload application loop (`acquire -> preprocess -> detect -> control -> actuate/publish`), with the first end-to-end payload integration test.

**Architecture:** A new module `flight/payload/app.py` defines `PayloadApp`, a frozen holder of injected services (sensor, gimbal, detector, controller, bus, clock, calibration, configs). `PayloadApp.process_frame()` runs one raw frame end-to-end; `PayloadApp.run()` is the thin acquisition loop. All decision logic stays in the pure `PayloadController` (Phase 5d); this module owns only I/O, sequencing, and message construction. Preprocessing runs co-located (no queue), honoring the co-location invariant. This faithfully reproduces the legacy `imaging` + `_run_inference_process` + `controller` processes (`src/pact/ops/main.py`, `src/pact/imaging/process.py`, `src/pact/controller/process.py`) collapsed into a single subsystem app.

**Tech Stack:** Python 3.14, numpy, frozen dataclasses, `Result[T, E]`, `Protocol`-based HAL, typed `MessageBus`, injected `Clock`. mypy --strict, ruff (line-length 100), import-linter, pytest.

---

## Context the implementer needs

**Faithful-migration source (the legacy loop being collapsed):**
- `src/pact/ops/main.py:89-236` `_run_inference_process()` — radiometric correction -> band selection -> quality flags -> build `ProcessedFrameMsg` -> inference. Note: the legacy did NOT crop (`crop_origin_px=(0, 0)`, `scale_factor=1.0`); preserve that.
- `src/pact/controller/process.py:123-216` — the per-frame control loop, already fully encapsulated by `PayloadController.step()` from Phase 5d.

**Migrated flight modules this app composes (verified signatures):**
- `flight.hal.interfaces` — `ImagingSensor.acquire_frame() -> Result[RawFrameMsg, FaultCode]`, `start_acquisition()/stop_acquisition() -> Result[None, FaultCode]`; `GimbalActuator.send_command(GimbalCommandMsg) -> Result[None, FaultCode]`, `read_position() -> Result[GimbalPosition, FaultCode]`.
- `flight.payload.preprocess` — `apply_calibration(raw, cal) -> Result[np.ndarray, FaultCode]`; `select_bands(raw, band_names: tuple[str, ...]) -> np.ndarray`; `compute_quality_flags(bands, exposure_us, utc_timestamp, cfg) -> frozenset[FrameUsabilityTag]`; `RadiometricCalibration(dark_frame, flat_field)`.
- `flight.payload.model` — `DetectorBackend.detect(ProcessedFrameMsg) -> Result[InferenceResultMsg, FaultCode]`; `ScriptedDetector(prob_mask, confidence_gate, min_blob_area_px, model_version)`.
- `flight.payload.control` — `PayloadController.from_config(ControllerConfig)`, `.initial_state() -> ControlState`, `.step(ControlState, InferenceResultMsg, now: float) -> tuple[ControlState, GimbalCommandMsg | None, list[TelemetryEventMsg]]`.
- `flight.libs.bus` — `MessageBus.publish(object)`, `.subscribe(type[T]) -> Subscription[T]`; `Subscription.empty()/get_nowait()`.
- `flight.libs.time` — `Clock.monotonic_s() -> float`, `.wall_clock_iso() -> str`; `ManualClock`.
- `flight.libs.config` — `PactConfig`, `ControllerConfig`, `InferenceConfig`, `PreprocessingConfig`, `FaultConfig`.
- `flight.libs.messages` — `RawFrameMsg`, `ProcessedFrameMsg`, `InferenceResultMsg`, `GimbalCommandMsg`, `TelemetryEventMsg`, `FaultEventMsg`, `HeartbeatMsg`.
- `flight.libs.types` — `Ok`, `Err`, `Result`, `FaultCode`, `GimbalState`, `MessageType`.

**Key design decisions (do not deviate):**
1. **Clock source for the arbiter `now`.** The legacy used `time.time()` (Unix). The app passes `clock.monotonic_s()` instead. This is behavior-preserving: the arbiter consumes `now` only as interval/rate-limit *deltas* (arbiter.py `_rate_ok`, idle accumulation), never as an absolute epoch. Monotonic time is exactly what `Clock.monotonic_s()` is documented for. Message timestamps use `clock.wall_clock_iso()`.
2. **No crop in the loop.** Match the legacy: `crop_origin_px=(0, 0)`, `scale_factor=1.0`. `crop_to_roi`/`backproject_pixel` remain available pure functions, unused here.
3. **Identity calibration sized from config.** Built once via `build_identity_calibration(cfg.inference)`: zero dark frame, unit flat field, shape `(len(input_bands), input_height_px, input_width_px)`. Raw frames fed to the app must match this shape. This mirrors the legacy `_calib` placeholder.
4. **Co-location invariant.** Preprocessing is a plain function call inside `process_frame`; `ProcessedFrameMsg` is never published to the bus (avoids pickling). The bus carries `InferenceResultMsg`, `GimbalCommandMsg`, `TelemetryEventMsg`, `FaultEventMsg`, `HeartbeatMsg`.
5. **Import-linter.** `flight.payload.app` may import only `flight.hal.interfaces` and `flight.libs.*` (plus sibling `flight.payload.*`). It must NOT import `flight.hal.drivers_sim`/`drivers_real` (the `drivers-from-composition-roots-only` contract). The test imports sim drivers freely (tests are outside the `flight` package and not analyzed by import-linter).
6. **No `**kwargs`/`*args`.** All parameters explicit. Frozen dataclasses; `slots=True` only on the pure data struct `TickOutcome` (not on `PayloadApp`, which holds Protocol-typed services — consistent with `PayloadController` being frozen-without-slots in Phase 5d).

---

### Task 1: Create the payload app shell module

**Files:**
- Create: `packages/flight/src/flight/payload/app.py`

- [ ] **Step 1: Write the module**

Create `packages/flight/src/flight/payload/app.py` with exactly this content:

```python
"""Payload application shell: binds the HAL and the pure payload core into one loop.

Collapses the legacy imaging + inference + controller processes into a single
in-process payload app. Per frame: acquire a raw frame from the imaging sensor,
preprocess it co-located (radiometric correction -> band selection -> quality flags;
no queue round-trip, honoring the preprocessing co-location invariant), run the
swappable detector, step the pure PayloadController, then drive the gimbal HAL and
publish results onto the typed bus. All decision logic lives in PayloadController;
this module owns only I/O, sequencing, and message construction.

Contains:
  - build_identity_calibration: identity RadiometricCalibration sized from InferenceConfig
    (zero dark frame, unit flat field) -- a placeholder until real sensor characterization.
  - TickOutcome: per-frame result summary (frame id, fault code, command-issued flag,
    resulting gimbal state) used for telemetry and testing.
  - PayloadApp: frozen holder of injected services. from_config() assembles it from a
    PactConfig and concrete drivers; process_frame() runs one frame end-to-end; run()
    is the acquisition loop (emits heartbeats, publishes a fault on camera stall).

Non-obvious notes:
  - The arbiter `now` is sourced from Clock.monotonic_s() (it consumes `now` only as
    interval/rate-limit deltas); message timestamps use Clock.wall_clock_iso().
  - No crop is applied (crop_origin_px=(0, 0), scale_factor=1.0), matching the legacy
    inference process; raw frames must match the identity calibration shape.

Satisfies: REQ-AIML-COMP-001, REQ-AIML-COMP-002 (payload process orchestration),
           REQ-OPER-HIGH-002 (subsystem app loop).
"""

from __future__ import annotations

# stdlib
import threading
from dataclasses import dataclass

# third-party
import numpy as np

# internal
from flight.hal.interfaces import GimbalActuator, ImagingSensor
from flight.libs.bus import MessageBus
from flight.libs.config import FaultConfig, InferenceConfig, PactConfig, PreprocessingConfig
from flight.libs.messages import (
    FaultEventMsg,
    HeartbeatMsg,
    ProcessedFrameMsg,
    RawFrameMsg,
)
from flight.libs.time import Clock
from flight.libs.types import Err, FaultCode, GimbalState, MessageType, Ok
from flight.payload.control import ControlState, PayloadController
from flight.payload.model import DetectorBackend
from flight.payload.preprocess import (
    RadiometricCalibration,
    apply_calibration,
    compute_quality_flags,
    select_bands,
)


def build_identity_calibration(cfg: InferenceConfig) -> RadiometricCalibration:
    """Build an identity radiometric calibration sized to the inference config.

    Produces a zero dark frame and a unit flat field of shape
    (len(input_bands), input_height_px, input_width_px), so apply_calibration is a
    no-op. Placeholder until real per-pixel dark/flat frames are available from
    sensor characterization; raw frames fed to the payload must match this shape.

    Args:
        cfg: InferenceConfig supplying band count and input height/width in pixels.

    Returns:
        RadiometricCalibration with zero dark_frame and unit flat_field (float32).
    """
    channels = len(cfg.input_bands)
    shape = (channels, cfg.input_height_px, cfg.input_width_px)
    return RadiometricCalibration(
        dark_frame=np.zeros(shape, dtype=np.float32),  # np.ndarray[float32, (C, H, W)]
        flat_field=np.ones(shape, dtype=np.float32),  # np.ndarray[float32, (C, H, W)]
    )


@dataclass(frozen=True, slots=True)
class TickOutcome:
    """Summary of one payload cycle, returned by process_frame for telemetry/testing.

    Attributes:
        frame_id: The frame_id of the processed raw frame.
        fault: FaultCode if preprocessing or detection failed this frame, else None.
        command_issued: True if a GimbalCommandMsg was sent to the gimbal this frame.
        gimbal_state: The arbiter GimbalState after this frame.
    """

    frame_id: int
    fault: FaultCode | None
    command_issued: bool
    gimbal_state: GimbalState


@dataclass(frozen=True)
class PayloadApp:
    """Payload subsystem app: imperative shell around the pure payload core.

    Holds the injected HAL drivers, detector, pure controller, bus, clock,
    calibration, and the config slices needed for preprocessing and heartbeats.
    Frozen to prevent field reassignment; the held services are themselves mutable
    (consistent with the composition-root injection pattern).
    """

    sensor: ImagingSensor
    gimbal: GimbalActuator
    detector: DetectorBackend
    controller: PayloadController
    bus: MessageBus
    clock: Clock
    calib: RadiometricCalibration
    inference_cfg: InferenceConfig
    preprocessing_cfg: PreprocessingConfig
    fault_cfg: FaultConfig

    @staticmethod
    def from_config(
        cfg: PactConfig,
        sensor: ImagingSensor,
        gimbal: GimbalActuator,
        detector: DetectorBackend,
        bus: MessageBus,
        clock: Clock,
    ) -> PayloadApp:
        """Assemble a PayloadApp from a PactConfig and concrete injected services.

        Builds the pure PayloadController from cfg.controller and an identity
        calibration from cfg.inference; carries cfg.preprocessing and cfg.fault for
        the loop. The drivers, detector, bus, and clock are injected by the caller
        (the composition root chooses real vs sim).

        Args:
            cfg: Top-level PactConfig.
            sensor: ImagingSensor driver (sim or real).
            gimbal: GimbalActuator driver (sim or real).
            detector: DetectorBackend (ScriptedDetector or OnnxDetector).
            bus: The typed MessageBus to publish onto.
            clock: Injected Clock (RealClock in flight, ManualClock in tests).

        Returns:
            A fully constructed PayloadApp.
        """
        return PayloadApp(
            sensor=sensor,
            gimbal=gimbal,
            detector=detector,
            controller=PayloadController.from_config(cfg.controller),
            bus=bus,
            clock=clock,
            calib=build_identity_calibration(cfg.inference),
            inference_cfg=cfg.inference,
            preprocessing_cfg=cfg.preprocessing,
            fault_cfg=cfg.fault,
        )

    def process_frame(
        self,
        raw: RawFrameMsg,
        state: ControlState,
        now: float,
    ) -> tuple[ControlState, TickOutcome]:
        """Process one raw frame end-to-end: preprocess -> detect -> control -> actuate.

        Runs preprocessing co-located (no queue), then the detector, then the pure
        PayloadController. Publishes InferenceResultMsg and each arbiter
        TelemetryEventMsg; when a command is issued it is both sent to the gimbal HAL
        and published. On a preprocessing or detection fault the state is returned
        unchanged, a FaultEventMsg is published, and outcome.fault is set.

        Args:
            raw: Raw frame to process. raw.raw_bands must match the calibration shape
                (len(input_bands), input_height_px, input_width_px).
            state: ControlState carried from the previous frame.
            now: Monotonic seconds for the arbiter (interval/rate-limit deltas only).

        Returns:
            (new_state, outcome). new_state is unchanged on a fault before control.
        """
        raw_bands = np.asarray(raw.raw_bands, dtype=np.float32)  # np.ndarray[float32, (C, H, W)]

        calib_result = apply_calibration(raw_bands, self.calib)
        if isinstance(calib_result, Err):
            self._publish_fault(
                calib_result.error, f"radiometric calibration failed frame_id={raw.frame_id}"
            )
            return state, self._fault_outcome(raw.frame_id, calib_result.error, state)
        cal_bands = calib_result.value

        selected = select_bands(cal_bands, self.inference_cfg.input_bands)  # (4, H, W) float32
        quality_flags = compute_quality_flags(
            selected, raw.exposure_us, raw.timestamp_utc, self.preprocessing_cfg
        )
        processed = ProcessedFrameMsg(
            msg_type=MessageType.PROCESSED_FRAME,
            timestamp_utc=raw.timestamp_utc,
            frame_id=raw.frame_id,
            tensor=selected,
            quality_flags=quality_flags,
            crop_origin_px=(0, 0),
            scale_factor=1.0,
        )

        detect_result = self.detector.detect(processed)
        if isinstance(detect_result, Err):
            self._publish_fault(
                detect_result.error, f"detection failed frame_id={raw.frame_id}"
            )
            return state, self._fault_outcome(raw.frame_id, detect_result.error, state)
        inference = detect_result.value
        self.bus.publish(inference)

        new_state, command, telemetry = self.controller.step(state, inference, now)
        for event in telemetry:
            self.bus.publish(event)

        if command is not None:
            send_result = self.gimbal.send_command(command)
            if isinstance(send_result, Err):
                self._publish_fault(
                    send_result.error, f"gimbal command failed frame_id={raw.frame_id}"
                )
            self.bus.publish(command)

        outcome = TickOutcome(
            frame_id=raw.frame_id,
            fault=None,
            command_issued=command is not None,
            gimbal_state=new_state.arbiter.gimbal_state,
        )
        return new_state, outcome

    def run(self, stop_event: threading.Event) -> None:
        """Run the payload acquisition loop until stop_event is set.

        Starts acquisition, then repeatedly: emits a HeartbeatMsg every
        fault_cfg.watchdog_interval_s, acquires a frame, and processes it (publishing
        a FaultEventMsg on a camera stall). Stops acquisition on exit. Control state
        is threaded internally, starting from controller.initial_state().

        Args:
            stop_event: threading.Event; the loop exits cleanly once it is set.
        """
        self.sensor.start_acquisition()
        state = self.controller.initial_state()
        heartbeat_seq = 0
        last_heartbeat = self.clock.monotonic_s()
        try:
            while not stop_event.is_set():
                now = self.clock.monotonic_s()
                if now - last_heartbeat >= self.fault_cfg.watchdog_interval_s:
                    self.bus.publish(
                        HeartbeatMsg(
                            msg_type=MessageType.HEARTBEAT,
                            timestamp_utc=self.clock.wall_clock_iso(),
                            subsystem="payload",
                            sequence=heartbeat_seq,
                        )
                    )
                    heartbeat_seq += 1
                    last_heartbeat = now
                acq = self.sensor.acquire_frame()
                if isinstance(acq, Ok):
                    state, _outcome = self.process_frame(acq.value, state, now)
                else:
                    self._publish_fault(acq.error, "imaging sensor stall")
        finally:
            self.sensor.stop_acquisition()

    def _publish_fault(self, code: FaultCode, detail: str) -> None:
        """Publish a FaultEventMsg from the payload subsystem onto the bus.

        Args:
            code: The FaultCode to report.
            detail: Human-readable detail string for logging/telemetry.
        """
        self.bus.publish(
            FaultEventMsg(
                msg_type=MessageType.FAULT_EVENT,
                timestamp_utc=self.clock.wall_clock_iso(),
                fault_code=code,
                subsystem="payload",
                detail=detail,
            )
        )

    def _fault_outcome(
        self, frame_id: int, code: FaultCode, state: ControlState
    ) -> TickOutcome:
        """Build a TickOutcome for a frame that faulted before control ran.

        Args:
            frame_id: The frame_id that faulted.
            code: The FaultCode raised.
            state: The unchanged control state (its arbiter state is reported).

        Returns:
            A TickOutcome with command_issued=False and the prior gimbal state.
        """
        return TickOutcome(
            frame_id=frame_id,
            fault=code,
            command_issued=False,
            gimbal_state=state.arbiter.gimbal_state,
        )
```

- [ ] **Step 2: Verify it imports and type-checks**

Run: `uv run mypy packages/flight/src/flight/payload/app.py`
Expected: `Success: no issues found`. (If mypy reports an `np.asarray(object)` error, it is wrong — `flight/payload/model/detector.py:101` does the same with `frame.tensor: object` and passes; the broad numpy `asarray` overload accepts it. Do not add `# type: ignore`.)

- [ ] **Step 3: Commit**

```bash
git add packages/flight/src/flight/payload/app.py
git commit -m "feat(payload): add payload app shell (acquire->preprocess->detect->control)"
```

---

### Task 2: Payload app integration + unit tests

**Files:**
- Create: `packages/flight/tests/test_payload_app.py`

- [ ] **Step 1: Write the tests**

Create `packages/flight/tests/test_payload_app.py` with exactly this content:

```python
"""Integration tests for the payload application shell (acquire->...->command)."""

import threading

import numpy as np
from flight.hal.drivers_sim import SimGimbal, SimSensor
from flight.libs.bus import MessageBus
from flight.libs.config import PactConfig
from flight.libs.messages import GimbalCommandMsg, InferenceResultMsg, RawFrameMsg
from flight.libs.time import ManualClock
from flight.libs.types import GimbalState, MessageType, Ok
from flight.payload.app import PayloadApp, TickOutcome
from flight.payload.model import ScriptedDetector


def _raw_frame(frame_id: int) -> RawFrameMsg:
    """Build a (4, 256, 256) zero-band raw frame matching the identity calibration."""
    raw_bands = np.zeros((4, 256, 256), dtype=np.float32)  # np.ndarray[float32, (C, H, W)]
    return RawFrameMsg(
        msg_type=MessageType.RAW_FRAME,
        timestamp_utc="2026-06-01T00:00:00.000Z",
        frame_id=frame_id,
        raw_bands=raw_bands,
        exposure_us=1000.0,
        gain_db=0.0,
        gimbal_az_deg=0.0,
        gimbal_el_deg=0.0,
    )


def _plume_detector() -> ScriptedDetector:
    """Scripted detector whose mask yields one strong, stable central blob each frame."""
    mask = np.zeros((256, 256), dtype=np.float32)  # np.ndarray[float32, (H, W)]
    mask[100:150, 100:150] = 1.0
    return ScriptedDetector(mask, confidence_gate=0.55, min_blob_area_px=15)


def _build_app(detector: ScriptedDetector) -> tuple[PayloadApp, MessageBus, SimGimbal]:
    """Assemble a PayloadApp over sim drivers, the given detector, and a fresh bus."""
    cfg = PactConfig()
    bus = MessageBus()
    gimbal = SimGimbal()
    sensor = SimSensor([])  # frames are fed directly to process_frame in these tests
    app = PayloadApp.from_config(cfg, sensor, gimbal, detector, bus, ManualClock())
    return app, bus, gimbal


def test_persistent_plume_drives_gimbal_through_app() -> None:
    """A stable plume across frames drives the app to TRACKING and moves the gimbal."""
    app, bus, gimbal = _build_app(_plume_detector())
    cmd_sub = bus.subscribe(GimbalCommandMsg)
    inf_sub = bus.subscribe(InferenceResultMsg)

    state = app.controller.initial_state()
    outcomes: list[TickOutcome] = []
    now = 0.0
    for frame_id in range(1, 9):
        now += 1.0
        state, outcome = app.process_frame(_raw_frame(frame_id), state, now)
        outcomes.append(outcome)

    assert state.arbiter.gimbal_state is GimbalState.TRACKING
    assert any(o.command_issued for o in outcomes)
    assert not cmd_sub.empty()  # at least one gimbal command was published

    position = gimbal.read_position()
    assert isinstance(position, Ok)
    assert (position.value.az_deg, position.value.el_deg) != (0.0, 0.0)  # gimbal moved

    inference_count = 0
    while not inf_sub.empty():
        inf_sub.get_nowait()
        inference_count += 1
    assert inference_count == 8  # one InferenceResultMsg published per frame


def test_no_detection_publishes_inference_but_no_command() -> None:
    """With an empty mask, frames are inferred and published but no command is issued."""
    empty_detector = ScriptedDetector(
        np.zeros((256, 256), dtype=np.float32), confidence_gate=0.55, min_blob_area_px=15
    )
    app, bus, _gimbal = _build_app(empty_detector)
    cmd_sub = bus.subscribe(GimbalCommandMsg)

    state = app.controller.initial_state()
    now = 0.0
    for frame_id in range(1, 6):
        now += 1.0
        state, outcome = app.process_frame(_raw_frame(frame_id), state, now)
        assert outcome.command_issued is False

    assert state.arbiter.gimbal_state is GimbalState.IDLE
    assert cmd_sub.empty()


def test_run_loop_starts_and_stops_cleanly() -> None:
    """run() returns promptly when stop_event is pre-set, exercising acquisition glue."""
    app, bus, _gimbal = _build_app(_plume_detector())
    cmd_sub = bus.subscribe(GimbalCommandMsg)

    stop = threading.Event()
    stop.set()
    app.run(stop)  # start + stop acquisition, no frame processed

    assert cmd_sub.empty()
```

- [ ] **Step 2: Run the tests to verify they pass**

Run: `uv run pytest packages/flight/tests/test_payload_app.py -v`
Expected: 3 passed. (`test_persistent_plume_drives_gimbal_through_app` reaches TRACKING at frame 3 and issues rate-limited commands at frames 3/5/7; the LQR refinement moves the SimGimbal off the origin.)

If `inference_count` is not 8 or TRACKING is not reached, do NOT weaken the assertions — re-verify the mask region area (2500 px >= 15), confidence (1.0 >= 0.55), and that `now` advances 1.0 s per frame against `retarget_rate_limit_hz=0.5`. These mirror the passing Phase 5d controller test.

- [ ] **Step 3: Commit**

```bash
git add packages/flight/tests/test_payload_app.py
git commit -m "test(payload): add payload app shell integration tests"
```

---

### Task 3: Full gate sweep

**Files:** none (verification only).

- [ ] **Step 1: Run every CI gate, scoped to packages/**

Run each and confirm green:

```bash
uv run ruff check packages
uv run ruff format --check packages
uv run mypy packages
uv run lint-imports
uv run pytest packages -m "not e2e"
```

Expected:
- `ruff check packages` -> All checks passed!
- `ruff format --check packages` -> all files already formatted. If it reports `app.py` or `test_payload_app.py` would be reformatted, run `uv run ruff format packages`, then `git add` + commit with `style: ruff-format payload app shell`.
- `mypy packages` -> Success (now 87 source files).
- `lint-imports` -> Contracts: 7 kept, 0 broken. (Critical: confirms `flight.payload.app` does NOT import `flight.hal.drivers_sim`/`drivers_real`.)
- `pytest packages -m "not e2e"` -> 133 passed, 1 skipped (130 from Phase 5d + 3 new).

- [ ] **Step 2: Commit any formatting fix (only if Step 1 required one)**

```bash
git add packages/flight/src/flight/payload/app.py packages/flight/tests/test_payload_app.py
git commit -m "style: ruff-format payload app shell"
```

---

## HARD RULES for the implementer

- Touch ONLY `packages/flight/src/flight/payload/app.py` and `packages/flight/tests/test_payload_app.py`. Do NOT modify `src/pact/**` (additive migration — `src/pact` stays untouched), do NOT modify `flight/payload/__init__.py`, and do NOT stage the pre-existing dirty working-tree entries (`src/pact/fault/detector.py`, `tests/**`, `.idea/*`, `.claude/settings.local.json`, `.coverage`, `bash.exe.stackdump`).
- Commits are LOCAL only; do not push.
- Preserve the Preprocessing Co-Location Invariant: no `ProcessedFrameMsg` on the bus; preprocessing stays a plain call inside `process_frame`.
- If a gate fails, fix the cause; do not weaken a test assertion or add `# type: ignore` to pass.

## Self-Review (spec coverage)

- Acquire (`ImagingSensor`) -> `run()` loop. ✓
- Preprocess (calibration -> band select -> quality flags, co-located, no crop). ✓ `process_frame`.
- Infer (`DetectorBackend.detect`). ✓
- Control (`PayloadController.step`). ✓
- Gimbal command + bus publish + telemetry + fault + heartbeat. ✓
- First payload integration test (end-to-end tracking through the app). ✓ Task 2.
- Layering preserved (app depends only on HAL interfaces + libs). ✓ Task 3 `lint-imports`.
```
