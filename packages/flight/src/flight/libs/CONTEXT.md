# `libs` Subsystem Context

Non-obvious, cross-cutting context for `flight.libs` that is not derivable from the
individual files or their docstrings.

## Layering

- `libs` is the bottom layer. Within it: `types < messages` (messages import enums);
  `config`, `bus`, `time`, `telemetry`, `ccsds`, and `commands` are mutually independent
  (except `commands` imports `types` for `CommandId`/`ParamKind`/`Result`/`FaultCode`).
  Everything above the layer imports from here, never the reverse.
- Always import from the package roots (`flight.libs.types`, `flight.libs.messages`,
  `flight.libs.config`, `flight.libs.ccsds`, `flight.libs.commands`), never the inner
  submodules -- the internal split is meant to stay refactorable. The `__init__` re-exports
  are the contract.
- `flight.libs.ccsds` and `flight.libs.commands` live in `libs` (not in app packages) so
  both `iss_iface` (app) and `hal.drivers_real.station` (driver) can import them without
  a layering violation.

## ccsds (added 2026-06-13, ADR 0009)

- `flight.libs.ccsds` is the CCSDS 133.0-B-2 Space Packet codec (pure, stdlib only).
- `encode_packet(header, body) -> Result[bytes, FaultCode]` builds a framed packet:
  6-byte primary header (3 x 16-bit big-endian words) + body + 4-byte CRC-32 trailer.
- `decode_packet(raw) -> Result[(CcsdsHeader, body), FaultCode]` CRC-verifies and strips the
  frame; returns `Err(COMMAND_CRC_FAIL)` on truncation or CRC mismatch.
- `packet_length(primary_header) -> Result[int, FaultCode]` extracts the total framed packet
  size from the first 6 bytes -- used by the real driver to deframe a TCP byte stream.
- `compute_crc32` / `verify_crc32`: the CRC-32 (ISO-3309/zlib, `binascii.crc32 & 0xFFFFFFFF`)
  primitives. The CRC covers `header + body` (before the CRC bytes themselves); the CRC
  trailer is 4 bytes big-endian appended after the body.
- Never raises; always returns `Result`. The codec is transport-agnostic (no sockets).

## commands (added 2026-06-13, ADR 0009)

- `flight.libs.commands` is the typed command dictionary: the validation authority for inbound
  ground commands.
- `CommandSpec` / `ParamSpec`: frozen `@dataclass(slots=True, frozen=True)` describing one
  command (target subsystem, required params, hazard class) and one param (name + `ParamKind`).
- `COMMAND_DICTIONARY: dict[CommandId, CommandSpec]` maps every `CommandId` to its spec.
  Phase 6A commands: `PING` (no-param, target `iss_iface`), `NOOP` (no-param, target
  `iss_iface`), `SET_THERMAL_LIMIT` (param `limit_c: float`, target `thermal`).
- `lookup_command(command_id: str) -> Result[CommandSpec, FaultCode]`: resolves a wire string;
  `Err(COMMAND_INVALID)` on unknown IDs.
- `validate_command(spec, params) -> Result[None, FaultCode]`: checks the params dict exactly
  against the spec's `ParamSpec` list (exact key set + per-key primitive kind). Data-not-dispatch:
  no callable tables, no `getattr` -- pure iteration over declared fields.
- `bool` / `int` discrimination: `bool` is a subclass of `int` in Python; `validate_command`
  rejects a `bool` where `INT`/`FLOAT` is required and vice versa.
- `CommandSpec.hazardous=True` is unused in Phase 6A but present for the Phase 6B router.

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
- **`Band` enum** (added 2026-06-09): `BLUE/GREEN/RED/NIR` replaces the legacy `B2/B3/B4/B8`
  Sentinel-2 band IDs. Passbands approximate Sentinel-2 B2/B3/B4/B8 (490/560/665/842 nm) so
  the training dataset remains a valid domain; using sensor-vocabulary names decouples PACT's
  band names from the origin dataset.
- **`MosaicFrame`** (added 2026-06-09): raw-frame value type passed from the imaging HAL to
  the payload app. Fields: `timestamp_utc`, `frame_id`, `mosaic` (`np.ndarray[uint16, (H, W)]`
  raw 2x2-CFA plane), `exposure_us`, `gain_db`. NOT a bus message; frames are passed by direct
  call (co-location invariant).
- **Ingest fault codes** (added 2026-06-09): `CALIBRATION_INVALID` (any startup calibration
  integrity failure -- shape mismatch, checksum mismatch, missing file) and `FRAME_MALFORMED`
  (a per-frame geometry violation in demosaic or band selection).
- **`RawFrameMsg` removed** (2026-06-09, ADR 0007): there is no bus message for raw or
  separated band stacks. `MessageType.RAW_FRAME` is likewise removed. A live reference to either
  name -- an import, construction, or `MessageType` lookup -- is a bug; the only remaining
  mentions are the removal test (`tests/test_messages.py`), this ADR, and these CONTEXT notes.
- **`GimbalCommandMode` enum** (added 2026-06-11, ADR 0008): `RATE` / `ABSOLUTE` / `STOW` / `HOME`.
  Carried by the pure-core `GimbalRequest` (`flight.payload.gimbal.request`, not a bus message)
  and echoed in `GimbalCommandMsg`. `GIMBAL_FAULT` (a driver-level gimbal failure) was added to
  `FaultCode` and is in `SAFE_TRIGGERING_FAULTS`.
- **`GimbalCommandMsg` reshaped** (2026-06-11, ADR 0008): it is now a *telemetry record* of an
  issued command, not a command carrier. Fields: `mode: GimbalCommandMode`, `az_value_deg`,
  `el_value_deg` (rate for RATE, target angle for ABSOLUTE, 0 otherwise), `state`, `reason`. The
  old `az_delta_deg`/`el_delta_deg` delta fields are gone.
- **`InferenceResultMsg` crop fields** (2026-06-11, ADR 0008): gains `crop_origin_px: tuple[int,
  int]` and `scale_factor: float`, copied from the `ProcessedFrameMsg`, so the controller can
  back-project a tensor centroid to full-plane pixels for boresight-error math.
- **`GimbalPosition.timestamp_s`** (2026-06-11, ADR 0008): `read_position` now returns a
  monotonic-stamped pose; the encoder-runaway monitor needs the timestamp to compute measured rate.
- **New enums (2026-06-13, ADR 0009):**
  - `LinkState`: `AOS` / `LOS` -- station link acquisition state; used by `StationLink.link_state()`
    and carried in `LinkStateMsg`. Published by iss_iface each tick.
  - `AckStatus`: `ACCEPTED` / `REJECTED` -- outcome of one inbound command at ingress; carried in
    `CommandAckMsg`.
  - `CommandId`: the command dictionary's opcode keys (`PING`, `NOOP`, `SET_THERMAL_LIMIT`). Phase
    6B will add hazardous opcodes (`EXIT_SAFE`, gimbal slew, lock release).
  - `ParamKind`: `STR` / `INT` / `FLOAT` / `BOOL` -- the primitive kind a command parameter must
    be for dictionary validation.
- **New `MessageType` members (2026-06-13, ADR 0009):** `COMMAND_ACK`, `LINK_STATE`.
- **New `FaultCode` members (2026-06-13, ADR 0009):** `COMMAND_CRC_FAIL`, `COMMAND_AUTH_FAIL`,
  `COMMAND_SEQ_ERROR`, `COMMAND_INVALID` -- all log-and-continue (not SAFE-triggering).

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
- **`CommandAckMsg` (2026-06-13, ADR 0009):** Acknowledgement (positive or negative) for one
  inbound ground command. Fields: `msg_type`, `timestamp_utc`, `status: AckStatus`,
  `command_id: str`, `source: str`, `seq: int`, `fault_code: FaultCode`, `detail: str`.
  Emitted by iss_iface for every inbound packet (ingress accept/reject). On `REJECTED`,
  `fault_code` carries the reason; on `ACCEPTED` it is `FaultCode.NONE`.
- **`LinkStateMsg` (2026-06-13, ADR 0009):** Current station-link acquisition state published
  by iss_iface each tick. Fields: `msg_type`, `timestamp_utc`, `state: LinkState`.

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
- **`LinkConfig` (2026-06-13, ADR 0009):** Station data-link transport config -- TCP bind
  address/port for inbound commands, UDP host/port for outbound telemetry, socket timeout,
  and the CCSDS APIDs for TC and TM. The config loader validates APID range (0..0x7FF), port
  range (1..65535), and that `socket_timeout_s > 0`.
- **`CommandIngressConfig` (2026-06-13, ADR 0009):** HMAC key path, `require_auth` flag
  (false = skip auth, test/bench only), and `accepted_sources` tuple. The HMAC key bytes are
  loaded by the composition root, not by iss_iface itself.

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
