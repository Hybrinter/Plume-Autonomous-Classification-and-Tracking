# tools.analysis -- SIL telemetry capture / analysis / report (agent context)

Non-obvious, cross-cutting patterns for the SIL telemetry analysis tool. This is a **read-only
observability** layer: it drives the deterministic SIL (real flight apps over sim drivers) and
captures per-step datapoints; it never changes flight behavior.

## The recorder OWNS the stepping loop (on purpose)

`recorder.record_run` does not call `SilHarness.run_steps`. It re-implements the exact same loop
(advance clock, advance `now`, call `sim.sil.step_once`) so it can **hold the threaded state** that
two apps keep *off* `self`:

- `PayloadApp`'s `ControlState` (gimbal FSM, Kalman `x`/`P`, EMA, runaway/deadband strikes,
  commanded rates) is a `run()`-local threaded through `step_once`.
- `FaultApp`'s watchdog `entries` dict (per-subsystem `miss_count`, `last_heartbeat_time`) is the
  same.

Only the loop owner can observe these. The recorder threads them and exposes them on the
`SampleContext` (`payload_state`, `fault_entries`). The other eight apps DO hold readable mutable
state (`app.state` / `app.safety` / `app.lock_gate`) and are read directly off the wired
`SilSystem`.

## Capture is passive because the bus is fan-out

`MessageBus.subscribe` gives every subscriber its **own** queue. The recorder subscribes to all 19
message types and drains its own subscriptions each step -- this steals nothing from the apps. Never
"peek" an app's subscription; read the recorder's.

## One DeviceSample per step (seeded noise)

`SimGimbal.read_position()` redraws seeded encoder noise on **every** call, so the recorder reads
each driver exactly once per step into a frozen `DeviceSample` and every gimbal signal reads those
cached numbers (self-consistent + deterministic). The clean integrated pose / commanded rate / mode
/ replay cursors are read from the sim drivers' private fields **read-only** (observability only --
no mutation, no flight-behavior change).

## The single FSW touch

The only flight-software change is `MessageBus.queue_depth(message_type)` -- a read-only accessor
(satisfies `REQ-OBS-SIL-001`). Everything else is read-only introspection of existing state. Do not
add telemetry, control-flow, or any other flight change to support analysis.

## Some faults are injected, not organic (documented per scenario)

`step_once` synthesizes every monitored app's heartbeat each cycle, and the sim gimbal tracks
commands faithfully, so `WATCHDOG_EXPIRE` and `GIMBAL_RUNAWAY` are not reachable organically. Those
scenarios **inject the FDIR-input `FaultEventMsg`** (which the fault app routes through its SAFE
policy) and say so in the scenario description. `MODEL_CORRUPT` (raised by a failed activation) is in
`SAFE_TRIGGERING_FAULTS`, so the model-lifecycle run latches SAFE after the rollback -- expected.

## The signal registry

`datapoints.REGISTRY` is a tuple of frozen `Signal`s, each carrying a statically-typed `ExtractorFn`
(`Callable[[SampleContext], float | str]`) -- mirrors `tools.accept`'s typed-`Callable`-field
convention, not a getattr/dispatch-by-string table. A failed extractor -> NaN (numeric) or `""`
(categorical); the recorder catches, so extractors stay naive. Per-step event-count signals (those
whose title ends `/step` or `this step`, see `is_event_rate`) get a derived `<name>.cumulative`
running-total column in the recorder.

## Determinism

`ManualClock` + fixed scene seed + step-index time axis; the manifest carries **no** wall-clock
timestamp. Re-running a scenario reproduces byte-identical long/stats/manifest. Keep it that way (no
`datetime.now`, no `Math.random`, no unsorted dict iteration in emitted data).
