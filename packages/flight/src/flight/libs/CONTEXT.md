# `libs` Subsystem Context

Non-obvious, cross-cutting context for `flight.libs` that is not derivable from the
individual files or their docstrings.

## Layering

- `libs` is the bottom layer. Within it: `types < messages` (messages import enums);
  `config`, `bus`, `time`, `telemetry` are mutually independent and import nothing from
  `flight`. Everything above the layer imports from here, never the reverse.
- Always import from the package roots (`flight.libs.types`, `flight.libs.messages`,
  `flight.libs.config`), never the inner submodules -- the internal split is meant to stay
  refactorable. The `__init__` re-exports are the contract.

## types

- `Ok`/`Err` are `@dataclass(frozen=True)`, deliberately NOT `slots=True`, and retain the
  explicit `Generic[T]` / `Union` forms (with `noqa`). The in-code "Rust-idiomatic parity"
  comments are legacy -- the Rust migration is dropped (`docs/adr/0001-python-only-drop-rust.md`)
  -- but the explicit `Result[T, E]` shape is the stable public contract used everywhere, so do
  not churn it into PEP 695 / `type` syntax without a deliberate, repo-wide reason.
- Enum string value equals the member name for log readability -- EXCEPT the two integer-ish
  cases: `DownlinkPriority` uses ints `0..3` (lower == higher priority; consumed directly by
  `queue.PriorityQueue` via `.value`). It is `enum.Enum`, not `IntEnum`, so it never
  serializes as a bare int into CCSDS packets.

## messages

- Every message is frozen, with `msg_type: MessageType` first and `timestamp_utc: str`
  second -- this ordering is a convention relied on across subsystems, not enforced.
- Large numpy arrays (`raw_bands`, `tensor`, `mask`, `raw_frame`, `processed_tensor`) are
  typed `object`, not `np.ndarray`. Frozen dataclasses cannot enforce dtype/shape at
  construction; the `# np.ndarray[float32, (C, H, W)]` comments are the only spec. The
  producing subsystem must validate shape before publishing.
- `utc_now_iso()` and `RealClock.wall_clock_iso()` are duplicate implementations of the same
  `...mmmZ` format. The trailing `Z` (not `+00:00`) is load-bearing: `storage/writer.py`'s
  directory-name parser depends on it. Change both together if you change either.
- No schema versioning exists on any message -- a known gap for multi-day missions where
  producer/consumer versions could drift.

## bus

- Routing is by EXACT type: `publish(msg)` matches `type(msg)`, with no subclass or interface
  dispatch. A subscriber registered for a base type will not receive a subclass.
- `publish` puts the SAME object reference into every subscriber queue -- there is no copy.
  Combined with array fields typed `object`, a consumer that mutates a received array
  corrupts it for every other subscriber. Treat received messages as immutable.
- Transport is in-process `queue.Queue` (unit tests / single-process SIL). The queue factory
  is the intended swap point for a multiprocessing-backed transport; the public API stays.

## config

- Frozen per-subsystem dataclasses; subsystems receive their typed config, never read TOML.
- Field defaults MUST exactly match `config/default.toml`. Unlike the old `src/pact` tree,
  divergence here IS guarded: `packages/flight/tests/test_config_defaults.py` loads the TOML
  and asserts equality. TOML arrays load as lists, so tuple defaults are compared after
  list->tuple normalization -- keep array-like defaults as tuples.

## time

- Two distinct clock channels: `monotonic_s()` for intervals/timeouts/rate limits,
  `wall_clock_iso()` for message stamps. Never use one for the other.
- Time is injected (the `Clock` Protocol), never read inside pure logic. `ManualClock`
  advances monotonic time explicitly and lets wall-clock be set, making time deterministic in
  tests.

## telemetry

- `configure_logging()` reconfigures global structlog state and is meant to be called exactly
  once at process startup; `flight_mode` toggles JSON (downlink) vs console (dev) rendering.
- `get_logger(subsystem)` binds `subsystem`; the `event` field is the first positional arg by
  structlog convention -- both are required on every log entry per project rules.
