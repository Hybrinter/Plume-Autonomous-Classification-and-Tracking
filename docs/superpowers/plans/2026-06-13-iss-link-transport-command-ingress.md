# ISS Link Transport + Authenticated Command Ingress Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the inert `StationLink` stub into a real CCSDS byte-level link, and turn `iss_iface` from a verbatim transport bridge into the authenticated command-ingress front door (decode -> CRC -> sequence dedup -> HMAC auth -> typed-dictionary validation -> publish `CommandMsg` + always emit an ACK/NACK downlink event).

**Architecture:** The link (`StationLink` HAL) becomes a pure byte transport: raw CCSDS Space Packets in (TC over TCP) and out (TM over UDP), with AOS/LOS state. All framing, integrity, authentication, and validation move up into `iss_iface`, whose ingress is a pure core (decode/auth/parse/validate) wrapped by a thin shell that owns the bus, the clock, the HMAC key, and the per-source sequence state. A typed command dictionary in `flight.libs` is the validation authority. Every inbound packet produces exactly one `CommandAckMsg` (ACCEPTED or REJECTED); only validated commands also become `CommandMsg` on the bus (consumed by the existing self-filtering target apps, unchanged).

**Tech Stack:** Python 3.12, stdlib only (`socket`, `struct`, `binascii`, `hmac`, `hashlib`, `json`, `threading`), numpy/scipy/structlog already present. No new third-party deps.

**Scope note (decomposition of spec Section 6):** This is Phase **6A** of the ISS command & data path. It deliberately covers transport + ingress + ack only. Deferred to later phases, each its own plan:
- **6B** -- command router (core service) + layered authority + ARM/EXECUTE hazardous two-step + inhibit-at-actuation + `EXIT_SAFE`/manual-gimbal/lock-release (closes the SAFE-recovery stub).
- **6C** -- data system: core storage service + prioritized downlink manager + `StorageWriter`/`StorageReader` Protocols.
- **6D** -- model upload: chunked reassembly + stage/activate/rollback + `ModelDeployState`.

Source spec: `docs/superpowers/specs/2026-06-09-pact-flight-final-state-design.md` Section 6 (link transport + command ingress) and ADR #3. Baseline survey: this plan was written against a full source survey of the current code (2026-06-13).

---

## Design Decisions (read before starting)

These resolve the open questions the survey surfaced. They are binding for this plan.

1. **The link transports raw CCSDS packets (bytes), not `CommandMsg`.** The spec puts decode/CRC/dedup/HMAC/validate in `iss_iface`, so the link must hand up raw bytes. The `StationLink` Protocol changes from command-level (`receive_command -> CommandMsg`) to byte-level (`receive_packet -> bytes`, `send_packet(bytes)`), plus `link_state()` and `close()`.

2. **`iss_iface` owns CCSDS framing; the link is a dumb byte pipe.** `iss_iface` encodes/decodes packets via `flight.libs.ccsds`. The real driver only needs CCSDS *length-field awareness* to deframe a TCP byte stream into discrete packets (it does NOT verify CRC/HMAC -- that is ingress's job).

3. **Wire format.**
   - **Telecommand (TC, inbound):** `[6-byte CCSDS primary header, packet_type=1, apid=tc_apid] [body = JSON-bytes of {command_id, params, source, seq}] [HMAC-SHA256 tag, 32 bytes] [CRC-32 trailer, 4 bytes big-endian over header+body+tag]`. The CCSDS `data_length` field covers `body + tag + crc`.
   - **Telemetry (TM, outbound):** `[6-byte primary header, packet_type=0, apid=tm_apid] [body = JSON-bytes] [CRC-32 trailer]`. No HMAC on downlink (out of scope; ground trusts the link).
   - CRC-32 = `binascii.crc32(data) & 0xFFFFFFFF` (mirrors legacy; ISO-3309/zlib poly).

4. **`CommandMsg` stays the raw envelope (str `target`/`command_id`).** No breaking change to the existing self-filtering target apps (thermal/electrical). Typed validation is added *around* it: `command_id` strings must equal a `CommandId.value`, and `target` is **set canonically by `iss_iface` from the command dictionary** (the ground frame carries `command_id`+`params`+`source`+`seq`; the dictionary pins the target). The dictionary is the typed authority.

5. **Command dictionary is data, not dispatch.** `COMMAND_DICTIONARY: dict[CommandId, CommandSpec]` where `CommandSpec`/`ParamSpec` are frozen *data* dataclasses (target name, required params + kinds, `hazardous` flag). Validation iterates the spec's declared params -- pure data-driven checks, no callable tables / `getattr` (honors `.claude/rules/strong_typing.md`).

6. **ACK/NACK transport.** Any subsystem may publish a `CommandAckMsg` on the bus; `iss_iface` is the single egress -- it subscribes to `CommandAckMsg`, encodes each into a TM packet, and `send_packet`s it (AOS-gated). For 6A, `iss_iface` itself produces the ingress ACCEPTED/REJECTED ack for every inbound packet.

7. **AOS/LOS.** The link reports `LinkState.AOS`/`LinkState.LOS`. `iss_iface` only drains its downlink/ack subscriptions when AOS (so acks/telemetry wait through LOS in the subscription queue). `iss_iface` publishes a `LinkStateMsg` each tick.

8. **HMAC key is injected, not loaded in-app.** The composition roots (`flight.core.main`, `sim.sil.runner`) load the key bytes and pass them through `build_apps` -> `IssIfaceApp.from_config` (mirrors how `MosaicCalibration` is injected). Unit/SIL tests pass key bytes directly -- no temp files.

9. **New fault codes are log-and-continue (NOT SAFE-triggering).** A bad/spoofed/replayed command must NACK, never SAFE the vehicle. `COMMAND_CRC_FAIL`, `COMMAND_AUTH_FAIL`, `COMMAND_SEQ_ERROR`, `COMMAND_INVALID` go in the log-and-continue partition.

10. **Layering homes (no import-linter edits needed):** CCSDS codec and command dictionary live in `flight.libs` (below apps; importable by both `iss_iface` and the real driver). The ingress pure core lives in `flight/iss_iface/ingress/` (app-owned). The real driver may import `flight.libs.ccsds` (drivers may import libs).

---

## File Structure

**New files:**
- `packages/flight/src/flight/libs/ccsds/__init__.py` -- re-exports the codec.
- `packages/flight/src/flight/libs/ccsds/codec.py` -- CCSDS Space Packet encode/decode + CRC.
- `packages/flight/src/flight/libs/commands/__init__.py` -- re-exports the dictionary + validator.
- `packages/flight/src/flight/libs/commands/dictionary.py` -- `CommandSpec`/`ParamSpec`/`COMMAND_DICTIONARY` + `validate_command`.
- `packages/flight/src/flight/iss_iface/ingress/__init__.py` -- re-exports the ingress pipeline.
- `packages/flight/src/flight/iss_iface/ingress/pipeline.py` -- pure decode/auth/parse/validate functions + `IngressOutcome`.
- `packages/flight/tests/test_ccsds_codec.py`
- `packages/flight/tests/test_command_dictionary.py`
- `packages/flight/tests/test_iss_ingress_pipeline.py`
- `packages/flight/tests/test_real_station_link.py` (loopback-socket tests)
- `docs/adr/0009-iss-link-transport-command-ingress.md`

**Modified files:**
- `packages/flight/src/flight/libs/types/enums.py` (+ `LinkState`, `AckStatus`, `CommandId`, `ParamKind`; + `MessageType.{COMMAND_ACK,LINK_STATE}`; + 4 `FaultCode`s)
- `packages/flight/src/flight/libs/types/__init__.py` (re-exports)
- `packages/flight/src/flight/libs/messages/messages.py` (+ `CommandAckMsg`, `LinkStateMsg`)
- `packages/flight/src/flight/libs/messages/__init__.py` (re-exports)
- `packages/flight/src/flight/libs/config/config.py` (+ `LinkConfig`, `CommandIngressConfig`; + fields on `PactConfig`)
- `packages/flight/src/flight/libs/config/__init__.py` (re-exports)
- `config/default.toml` (+ `[link]`, `[command_ingress]` sections)
- `packages/flight/src/flight/core/config_loader.py` (map + validate new sections)
- `packages/flight/src/flight/core/composition.py` (`build_apps` gains `uplink_key`; pass to iss_iface)
- `packages/flight/src/flight/core/main.py` (construct `RealStationLink(cfg, clock)`; load key)
- `packages/flight/src/flight/hal/interfaces/station.py` (byte-level Protocol)
- `packages/flight/src/flight/hal/drivers_sim/station.py` (byte-level sim link)
- `packages/flight/src/flight/hal/drivers_real/station.py` (real TCP/UDP socket driver)
- `packages/flight/src/flight/iss_iface/app.py` (ingress front door)
- `packages/flight/src/flight/fault/policy.py` (partition the new fault codes)
- `packages/sim/src/sim/sil/runner.py` (`build_sil_system` inbound bytes + key)
- Tests: `test_enums.py`, `test_policy.py` (or equivalent), `test_config_defaults.py`, `test_messages.py`, `test_iss_iface_app.py`, `test_hal_interfaces.py`, `test_real_drivers.py`, `packages/sim/tests/test_sil_closed_loop.py`
- CONTEXT docs: `iss_iface/CONTEXT.md`, `hal/CONTEXT.md`, `libs/CONTEXT.md`, `sim/CONTEXT.md`
- `docs/adr/README.md`

---

## Conventions reminder (apply to every task)

- ASCII only; 100-char lines; imports grouped stdlib -> third-party -> internal.
- Every non-test class/function/method gets a full docstring (summary, typed inputs/outputs, non-obvious notes). Module docstrings cite `Satisfies: REQ-...`. Tests get a one-line docstring only.
- Data structs: `@dataclass(slots=True)`; messages follow the local `frozen=True` (no slots) convention with `msg_type` first, `timestamp_utc` second. Config holders are `frozen=True` (no slots).
- Library code returns `Result[T, E]`, never raises (only startup misconfig in `main`/`_validate` raises via `Err`). Narrow `Ok`/`Err` with `isinstance` before reading `.value`/`.error`.
- Enum string values mirror member names (except the existing int-valued `DownlinkPriority`).
- structlog with `subsystem` + `event` fields anywhere the app shell logs.
- **Per-commit gates (all five must pass, run from repo root):**
  `uv run pytest packages` ; `uv run ruff check packages` ; `uv run ruff format --check packages` ; `uv run mypy packages` ; `uv run lint-imports`
- Stage files by **explicit path** only (`git add <path> ...`); never `git add -A`/`.`. Never touch `src/pact/**`, top-level `tests/**`, `.idea/**`, `.claude/**`, `.coverage`, `bash.exe.stackdump`, `docs/superpowers/baseline/**`.
- Every commit message ends with the trailer: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
- Do NOT push; do NOT amend existing commits.

---

## Task 1: Types foundation -- enums, message-type discriminants, fault codes, policy partition

**Files:**
- Modify: `packages/flight/src/flight/libs/types/enums.py`
- Modify: `packages/flight/src/flight/libs/types/__init__.py`
- Modify: `packages/flight/src/flight/fault/policy.py`
- Test: `packages/flight/tests/test_enums.py`, the policy test (find it: `packages/flight/tests/test_policy.py` or `test_fault_policy.py`)

- [ ] **Step 1: Write failing enum tests**

In `packages/flight/tests/test_enums.py`, add name-mirror smoke tests following the existing pattern:

```python
def test_link_state_values_mirror_names() -> None:
    """LinkState string values mirror member names."""
    for member in LinkState:
        assert member.value == member.name


def test_ack_status_values_mirror_names() -> None:
    """AckStatus string values mirror member names."""
    for member in AckStatus:
        assert member.value == member.name


def test_command_id_values_mirror_names() -> None:
    """CommandId string values mirror member names."""
    for member in CommandId:
        assert member.value == member.name


def test_param_kind_values_mirror_names() -> None:
    """ParamKind string values mirror member names."""
    for member in ParamKind:
        assert member.value == member.name


def test_new_message_types_present() -> None:
    """The command-ack and link-state discriminants exist."""
    assert MessageType.COMMAND_ACK.value == "COMMAND_ACK"
    assert MessageType.LINK_STATE.value == "LINK_STATE"


def test_new_command_fault_codes_present() -> None:
    """The command-ingress fault codes exist."""
    for name in ("COMMAND_CRC_FAIL", "COMMAND_AUTH_FAIL", "COMMAND_SEQ_ERROR", "COMMAND_INVALID"):
        assert FaultCode[name].value == name
```

Add the new names to the test module's imports from `flight.libs.types`.

- [ ] **Step 2: Run -- expect ImportError / AttributeError**

Run: `uv run pytest packages/flight/tests/test_enums.py -q`
Expected: FAIL (names not defined).

- [ ] **Step 3: Add the enums and members in `enums.py`**

Add four new `MessageType` members at the end of that enum (keep them grouped after `UPLINK_CHUNK`):

```python
    COMMAND_ACK = "COMMAND_ACK"
    LINK_STATE = "LINK_STATE"
```

Add four new `FaultCode` members at the end of `FaultCode` (after `GIMBAL_FAULT`):

```python
    COMMAND_CRC_FAIL = "COMMAND_CRC_FAIL"
    COMMAND_AUTH_FAIL = "COMMAND_AUTH_FAIL"
    COMMAND_SEQ_ERROR = "COMMAND_SEQ_ERROR"
    COMMAND_INVALID = "COMMAND_INVALID"
```

Add four new enum classes (place them near the other small enums; each needs a class docstring with the name-mirror note + a `Satisfies:` reference where applicable):

```python
class LinkState(enum.Enum):
    """Station link acquisition state. AOS = link up (drain downlink), LOS = link down.

    String values mirror member names (log readability convention). Satisfies: REQ-COMM-HIGH-001.
    """

    AOS = "AOS"  # acquisition of signal: contact established, downlink may drain
    LOS = "LOS"  # loss of signal: no contact, hold downlink


class AckStatus(enum.Enum):
    """Outcome of a single inbound command at ingress.

    String values mirror member names (log readability convention). Satisfies: REQ-COMM-HIGH-004.
    """

    ACCEPTED = "ACCEPTED"  # decoded, authenticated, and validated; CommandMsg published
    REJECTED = "REJECTED"  # failed CRC / auth / sequence / dictionary validation; no CommandMsg


class CommandId(enum.Enum):
    """The command dictionary's opcode keys (per-command schema lives in flight.libs.commands).

    String values mirror member names (log readability convention). Satisfies: REQ-COMM-HIGH-003.
    """

    PING = "PING"  # liveness check; non-hazardous; no params
    SET_THERMAL_LIMIT = "SET_THERMAL_LIMIT"  # non-hazardous; param limit_c: float
    NOOP = "NOOP"  # accepted no-op; non-hazardous; no params


class ParamKind(enum.Enum):
    """Primitive kind a command parameter must be, for dictionary validation.

    String values mirror member names (log readability convention). Satisfies: REQ-COMM-HIGH-003.
    """

    STR = "STR"
    INT = "INT"
    FLOAT = "FLOAT"
    BOOL = "BOOL"
```

Update the `enums.py` module docstring `Includes:`/`Contains:` list to mention the new enums.

> NOTE: `CommandId` members here are the minimal set 6A needs to exercise the ingress path (a no-param non-hazardous command `PING`/`NOOP` and a one-param command `SET_THERMAL_LIMIT` targeting the existing thermal app). Phase 6B adds the hazardous ones (`EXIT_SAFE`, gimbal slew, lock release).

- [ ] **Step 4: Re-export from `types/__init__.py`**

Add `AckStatus`, `CommandId`, `LinkState`, `ParamKind` to both the import block and `__all__` (keep alphabetical to match current ordering).

- [ ] **Step 5: Run enum tests -- expect PASS**

Run: `uv run pytest packages/flight/tests/test_enums.py -q`
Expected: PASS.

- [ ] **Step 6: Partition the new fault codes in policy + test**

The four new codes are NOT SAFE-triggering. They are already excluded from `SAFE_TRIGGERING_FAULTS` (since that set is an explicit allow-list), so `decide_mode_change` already returns `None` for them. But the policy docstring enumerates the log-and-continue set -- update it. In `packages/flight/src/flight/fault/policy.py`, extend the module docstring's `log-and-continue = {...}` clause to include the four new codes:

```
log-and-continue = {NONE, INFERENCE_TIMEOUT, STORAGE_FULL, COMM_TIMEOUT,
COMMAND_CRC_FAIL, COMMAND_AUTH_FAIL, COMMAND_SEQ_ERROR, COMMAND_INVALID}.
```

Find the policy test (e.g. `packages/flight/tests/test_policy.py`). If it asserts an exhaustive partition (every `FaultCode` is either SAFE-triggering or explicitly log-and-continue), add the four new codes to its non-SAFE expectation. Add a focused test:

```python
def test_command_ingress_faults_do_not_trigger_safe() -> None:
    """Command CRC/auth/seq/validation faults are annunciated, never SAFE the vehicle."""
    for code in (
        FaultCode.COMMAND_CRC_FAIL,
        FaultCode.COMMAND_AUTH_FAIL,
        FaultCode.COMMAND_SEQ_ERROR,
        FaultCode.COMMAND_INVALID,
    ):
        event = FaultEventMsg(
            msg_type=MessageType.FAULT_EVENT,
            timestamp_utc="2026-01-01T00:00:00.000Z",
            fault_code=code,
            subsystem="iss_iface",
            detail="test",
        )
        assert decide_mode_change(event, "2026-01-01T00:00:00.000Z") is None
```

- [ ] **Step 7: Run full gates, then commit**

Run all five gates. Expected: PASS.

```
git add packages/flight/src/flight/libs/types/enums.py packages/flight/src/flight/libs/types/__init__.py packages/flight/src/flight/fault/policy.py packages/flight/tests/test_enums.py packages/flight/tests/test_policy.py
git commit -m "feat(libs): add link/ack/command enums + command-ingress fault codes"
```
(Adjust the policy test path to the real filename.)

---

## Task 2: ACK and link-state messages

**Files:**
- Modify: `packages/flight/src/flight/libs/messages/messages.py`
- Modify: `packages/flight/src/flight/libs/messages/__init__.py`
- Test: `packages/flight/tests/test_messages.py`

- [ ] **Step 1: Write failing message tests**

In `packages/flight/tests/test_messages.py`, add:

```python
def test_command_ack_msg_fields() -> None:
    """CommandAckMsg carries the ack status and command correlation handles."""
    ack = CommandAckMsg(
        msg_type=MessageType.COMMAND_ACK,
        timestamp_utc="2026-01-01T00:00:00.000Z",
        status=AckStatus.ACCEPTED,
        command_id="PING",
        source="ground",
        seq=1,
        fault_code=FaultCode.NONE,
        detail="",
    )
    assert ack.status is AckStatus.ACCEPTED
    assert ack.command_id == "PING"
    assert ack.seq == 1


def test_link_state_msg_fields() -> None:
    """LinkStateMsg carries the current AOS/LOS state."""
    msg = LinkStateMsg(
        msg_type=MessageType.LINK_STATE,
        timestamp_utc="2026-01-01T00:00:00.000Z",
        state=LinkState.AOS,
    )
    assert msg.state is LinkState.AOS
```

Add the new symbols to the test imports.

- [ ] **Step 2: Run -- expect ImportError**

Run: `uv run pytest packages/flight/tests/test_messages.py -q` -> FAIL.

- [ ] **Step 3: Add the message dataclasses**

In `messages.py`, add the enum imports (`AckStatus`, `LinkState` from `flight.libs.types`) to the existing types import line, then add (follow the local `frozen=True`, msg_type-first, timestamp-second convention):

```python
@dataclass(frozen=True)
class CommandAckMsg:
    """Acknowledgement (positive or negative) for one inbound ground command.

    Emitted by iss_iface for every inbound packet (ingress accept/reject) and by target
    apps/services on execution (Phase 6B). Correlates back to the originating command via
    (source, seq, command_id). On REJECTED, fault_code carries the reason; on ACCEPTED it
    is FaultCode.NONE.
    """

    msg_type: MessageType  # must be MessageType.COMMAND_ACK
    timestamp_utc: str  # ISO 8601, millisecond precision
    status: AckStatus  # ACCEPTED or REJECTED
    command_id: str  # echoed opcode string ("" if the body was unparseable)
    source: str  # echoed command origin
    seq: int  # echoed per-source sequence number (-1 if unparseable)
    fault_code: FaultCode  # NONE on ACCEPTED; the reject reason otherwise
    detail: str  # human-readable reason / context


@dataclass(frozen=True)
class LinkStateMsg:
    """Current station-link acquisition state, published by iss_iface each tick."""

    msg_type: MessageType  # must be MessageType.LINK_STATE
    timestamp_utc: str  # ISO 8601, millisecond precision
    state: LinkState  # AOS (link up) or LOS (link down)
```

- [ ] **Step 4: Re-export**

Add `CommandAckMsg`, `LinkStateMsg` to the `from flight.libs.messages.messages import (...)` block and `__all__` in `messages/__init__.py` (keep alphabetical).

- [ ] **Step 5: Run -- expect PASS**

Run: `uv run pytest packages/flight/tests/test_messages.py -q` -> PASS.

- [ ] **Step 6: Gates + commit**

```
git add packages/flight/src/flight/libs/messages/messages.py packages/flight/src/flight/libs/messages/__init__.py packages/flight/tests/test_messages.py
git commit -m "feat(libs): add CommandAckMsg + LinkStateMsg bus messages"
```

---

## Task 3: CCSDS Space Packet codec (`flight.libs.ccsds`)

**Files:**
- Create: `packages/flight/src/flight/libs/ccsds/__init__.py`
- Create: `packages/flight/src/flight/libs/ccsds/codec.py`
- Test: `packages/flight/tests/test_ccsds_codec.py`

Port the bit layout from the legacy `src/pact/comms/ccsds.py` (reference only) but: return `Result[..., FaultCode]` (never raise), and embed a CRC-32 trailer so packets are self-validating.

- [ ] **Step 1: Write failing codec tests**

`packages/flight/tests/test_ccsds_codec.py`:

```python
"""CCSDS Space Packet codec round-trip and integrity tests."""

from flight.libs.ccsds import (
    CcsdsHeader,
    compute_crc32,
    decode_packet,
    encode_packet,
    packet_length,
)
from flight.libs.types import Err, FaultCode, Ok


def test_encode_decode_round_trip() -> None:
    header = CcsdsHeader(packet_type=1, apid=0x01, sequence_count=5)
    body = b"hello-command"
    encoded = encode_packet(header, body)
    assert isinstance(encoded, Ok)
    decoded = decode_packet(encoded.value)
    assert isinstance(decoded, Ok)
    out_header, out_body = decoded.value
    assert out_header.packet_type == 1
    assert out_header.apid == 0x01
    assert out_header.sequence_count == 5
    assert out_body == body


def test_decode_rejects_crc_corruption() -> None:
    header = CcsdsHeader(packet_type=0, apid=0x02, sequence_count=1)
    encoded = encode_packet(header, b"science").value
    corrupted = bytearray(encoded)
    corrupted[8] ^= 0xFF  # flip a body byte; CRC trailer no longer matches
    result = decode_packet(bytes(corrupted))
    assert isinstance(result, Err)
    assert result.error is FaultCode.COMMAND_CRC_FAIL


def test_decode_rejects_truncated() -> None:
    assert isinstance(decode_packet(b"\x00\x00\x00"), Err)


def test_encode_rejects_out_of_range_apid() -> None:
    result = encode_packet(CcsdsHeader(packet_type=1, apid=0x800, sequence_count=0), b"x")
    assert isinstance(result, Err)
    assert result.error is FaultCode.COMMAND_INVALID


def test_packet_length_reads_total_size_from_header() -> None:
    encoded = encode_packet(CcsdsHeader(packet_type=1, apid=1, sequence_count=0), b"abcd").value
    length = packet_length(encoded[:6])
    assert isinstance(length, Ok)
    assert length.value == len(encoded)


def test_compute_crc32_matches_binascii() -> None:
    import binascii

    assert compute_crc32(b"abc") == (binascii.crc32(b"abc") & 0xFFFFFFFF)
```

- [ ] **Step 2: Run -- expect ImportError**

Run: `uv run pytest packages/flight/tests/test_ccsds_codec.py -q` -> FAIL.

- [ ] **Step 3: Implement the codec**

`packages/flight/src/flight/libs/ccsds/codec.py`:

```python
"""CCSDS 133.0-B-2 Space Packet codec with a CRC-32 integrity trailer (pure, stdlib).

Encodes/decodes the 6-byte primary header (big-endian, three 16-bit words) and appends a
4-byte CRC-32 trailer over the whole packet so a decoded packet is self-validating. The
codec is transport-agnostic: it never touches sockets. iss_iface uses it to frame/deframe
command (TC, packet_type=1) and telemetry (TM, packet_type=0) packets; the real station
driver uses packet_length() to deframe a TCP byte stream into discrete packets.

Bit layout (mirrors the standard): word1 = version(3, 0) | type(1) | sec_hdr(1, 0) |
apid(11); word2 = seq_flags(2, 0b11 standalone) | seq_count(14); word3 = data_length(16)
= (len(body) + CRC_TRAILER_SIZE) - 1. CRC-32 = binascii.crc32 & 0xFFFFFFFF (ISO-3309/zlib).

Contains:
  - CcsdsHeader: decoded primary-header fields used by callers.
  - compute_crc32 / verify_crc32: the integrity primitive.
  - encode_packet: header + body -> framed bytes with CRC trailer (Result, never raises).
  - decode_packet: framed bytes -> (header, body), CRC-verified (Result).
  - packet_length: total packet size from the first 6 header bytes (for stream deframing).

Satisfies: REQ-COMM-HIGH-002.
"""

from __future__ import annotations

# stdlib
import binascii
import struct
from dataclasses import dataclass

# internal
from flight.libs.types import Err, FaultCode, Ok, Result

CCSDS_PRIMARY_HEADER_SIZE = 6
CRC_TRAILER_SIZE = 4
APID_MAX = 0x7FF
SEQ_COUNT_MAX = 0x3FFF
_SEQ_FLAGS_STANDALONE = 0b11


@dataclass(slots=True)
class CcsdsHeader:
    """Decoded CCSDS primary-header fields the caller cares about.

    Args/fields:
        packet_type: 0 = telemetry (TM), 1 = telecommand (TC).
        apid: 11-bit application process identifier (0..0x7FF).
        sequence_count: 14-bit per-APID packet sequence count (0..0x3FFF).
    """

    packet_type: int
    apid: int
    sequence_count: int


def compute_crc32(data: bytes) -> int:
    """Return the unsigned CRC-32 (ISO-3309 / zlib) of data."""
    return binascii.crc32(data) & 0xFFFFFFFF


def verify_crc32(data: bytes, expected: int) -> bool:
    """Return True iff compute_crc32(data) equals expected (masked to 32 bits)."""
    return compute_crc32(data) == (expected & 0xFFFFFFFF)


def encode_packet(header: CcsdsHeader, body: bytes) -> Result[bytes, FaultCode]:
    """Frame body into a CCSDS Space Packet with a CRC-32 trailer.

    Args:
        header: The primary-header fields (type / apid / sequence_count).
        body: The packet data field (already includes any HMAC tag for TC).

    Returns:
        Ok(framed bytes) = primary header + body + CRC-32(header+body), or
        Err(FaultCode.COMMAND_INVALID) if a header field is out of range or body is empty.
    """
    if not (0 <= header.apid <= APID_MAX):
        return Err(FaultCode.COMMAND_INVALID)
    if not (0 <= header.sequence_count <= SEQ_COUNT_MAX):
        return Err(FaultCode.COMMAND_INVALID)
    if header.packet_type not in (0, 1) or len(body) == 0:
        return Err(FaultCode.COMMAND_INVALID)

    data_length = len(body) + CRC_TRAILER_SIZE - 1
    if data_length > 0xFFFF:
        return Err(FaultCode.COMMAND_INVALID)

    word1 = (0 << 13) | ((header.packet_type & 0x01) << 12) | (0 << 11) | (header.apid & APID_MAX)
    word2 = (_SEQ_FLAGS_STANDALONE << 14) | (header.sequence_count & SEQ_COUNT_MAX)
    word3 = data_length & 0xFFFF
    primary = struct.pack(">HHH", word1, word2, word3)
    frame = primary + body
    return Ok(frame + struct.pack(">I", compute_crc32(frame)))


def decode_packet(raw: bytes) -> Result[tuple[CcsdsHeader, bytes], FaultCode]:
    """Decode and CRC-verify a framed CCSDS Space Packet.

    Args:
        raw: The complete framed packet (primary header + body + CRC trailer).

    Returns:
        Ok((header, body)) on success, Err(FaultCode.COMMAND_CRC_FAIL) on a length or CRC
        violation (truncated, inconsistent data_length, or CRC mismatch).
    """
    if len(raw) < CCSDS_PRIMARY_HEADER_SIZE + CRC_TRAILER_SIZE:
        return Err(FaultCode.COMMAND_CRC_FAIL)
    frame, crc_bytes = raw[:-CRC_TRAILER_SIZE], raw[-CRC_TRAILER_SIZE:]
    (expected_crc,) = struct.unpack(">I", crc_bytes)
    if not verify_crc32(frame, expected_crc):
        return Err(FaultCode.COMMAND_CRC_FAIL)

    word1, word2, word3 = struct.unpack(">HHH", frame[:CCSDS_PRIMARY_HEADER_SIZE])
    packet_type = (word1 >> 12) & 0x01
    apid = word1 & APID_MAX
    sequence_count = word2 & SEQ_COUNT_MAX
    body = frame[CCSDS_PRIMARY_HEADER_SIZE:]
    if (word3 & 0xFFFF) != len(body) + CRC_TRAILER_SIZE - 1:
        return Err(FaultCode.COMMAND_CRC_FAIL)
    return Ok((CcsdsHeader(packet_type=packet_type, apid=apid, sequence_count=sequence_count), body))


def packet_length(primary_header: bytes) -> Result[int, FaultCode]:
    """Return total framed-packet size from the first 6 header bytes (for stream deframing).

    Args:
        primary_header: At least the 6-byte CCSDS primary header.

    Returns:
        Ok(total bytes = 6 + (data_length + 1)) where the trailing CRC is included in
        data_length, or Err(FaultCode.COMMAND_CRC_FAIL) if fewer than 6 bytes are given.
    """
    if len(primary_header) < CCSDS_PRIMARY_HEADER_SIZE:
        return Err(FaultCode.COMMAND_CRC_FAIL)
    (_, _, word3) = struct.unpack(">HHH", primary_header[:CCSDS_PRIMARY_HEADER_SIZE])
    return Ok(CCSDS_PRIMARY_HEADER_SIZE + (word3 & 0xFFFF) + 1)
```

`packages/flight/src/flight/libs/ccsds/__init__.py`:

```python
"""CCSDS Space Packet codec (see flight.libs.ccsds.codec)."""

from flight.libs.ccsds.codec import (
    CcsdsHeader,
    compute_crc32,
    decode_packet,
    encode_packet,
    packet_length,
    verify_crc32,
)

__all__ = [
    "CcsdsHeader",
    "compute_crc32",
    "decode_packet",
    "encode_packet",
    "packet_length",
    "verify_crc32",
]
```

- [ ] **Step 4: Run -- expect PASS**

Run: `uv run pytest packages/flight/tests/test_ccsds_codec.py -q` -> PASS.

- [ ] **Step 5: Gates + commit**

```
git add packages/flight/src/flight/libs/ccsds/__init__.py packages/flight/src/flight/libs/ccsds/codec.py packages/flight/tests/test_ccsds_codec.py
git commit -m "feat(libs): add CCSDS Space Packet codec with CRC-32 trailer"
```

---

## Task 4: Typed command dictionary + validator (`flight.libs.commands`)

**Files:**
- Create: `packages/flight/src/flight/libs/commands/__init__.py`
- Create: `packages/flight/src/flight/libs/commands/dictionary.py`
- Test: `packages/flight/tests/test_command_dictionary.py`

- [ ] **Step 1: Write failing tests**

`packages/flight/tests/test_command_dictionary.py`:

```python
"""Command dictionary lookup + parameter-schema validation tests."""

from flight.libs.commands import COMMAND_DICTIONARY, lookup_command, validate_command
from flight.libs.types import CommandId, Err, FaultCode, Ok


def test_every_command_id_has_a_spec() -> None:
    for command_id in CommandId:
        assert command_id in COMMAND_DICTIONARY


def test_lookup_known_command() -> None:
    result = lookup_command("PING")
    assert isinstance(result, Ok)
    assert result.value.command_id is CommandId.PING


def test_lookup_unknown_command_rejected() -> None:
    result = lookup_command("NOT_A_COMMAND")
    assert isinstance(result, Err)
    assert result.error is FaultCode.COMMAND_INVALID


def test_validate_accepts_correct_params() -> None:
    spec = lookup_command("SET_THERMAL_LIMIT").value
    assert isinstance(validate_command(spec, {"limit_c": 70.0}), Ok)


def test_validate_rejects_missing_param() -> None:
    spec = lookup_command("SET_THERMAL_LIMIT").value
    result = validate_command(spec, {})
    assert isinstance(result, Err)
    assert result.error is FaultCode.COMMAND_INVALID


def test_validate_rejects_wrong_type() -> None:
    spec = lookup_command("SET_THERMAL_LIMIT").value
    assert isinstance(validate_command(spec, {"limit_c": "hot"}), Err)


def test_validate_rejects_unexpected_param() -> None:
    spec = lookup_command("PING").value
    assert isinstance(validate_command(spec, {"extra": 1}), Err)
```

- [ ] **Step 2: Run -- expect ImportError**

Run: `uv run pytest packages/flight/tests/test_command_dictionary.py -q` -> FAIL.

- [ ] **Step 3: Implement the dictionary + validator**

`packages/flight/src/flight/libs/commands/dictionary.py`:

```python
"""Typed command dictionary: the validation authority for inbound ground commands (pure).

Maps each CommandId to a CommandSpec carrying its canonical target subsystem, its required
parameter schema (name + primitive kind), and whether it is hazardous (ARM/EXECUTE gated --
enforced by the command router in Phase 6B). Validation is data-driven iteration over the
spec's declared params: no callable dispatch tables and no getattr (honors the strong-typing
rule). The dictionary lives in flight.libs so both iss_iface and the composition root can
import it without a layering violation.

Contains:
  - ParamSpec / CommandSpec: frozen data describing one parameter / one command.
  - COMMAND_DICTIONARY: the CommandId -> CommandSpec registry.
  - lookup_command: resolve a wire command_id string to its spec (Result).
  - validate_command: check a params dict against a spec's schema (Result).

Satisfies: REQ-COMM-HIGH-003.
"""

from __future__ import annotations

# stdlib
from dataclasses import dataclass

# internal
from flight.libs.types import CommandId, Err, FaultCode, Ok, ParamKind, Result

_KIND_TO_TYPE: dict[ParamKind, type] = {
    ParamKind.STR: str,
    ParamKind.INT: int,
    ParamKind.FLOAT: float,
    ParamKind.BOOL: bool,
}


@dataclass(slots=True, frozen=True)
class ParamSpec:
    """One required command parameter: its key and primitive kind.

    Note: ParamKind.FLOAT accepts int or float (ints widen to float); ParamKind.INT and
    ParamKind.BOOL are exact (bool is rejected where INT/FLOAT is required and vice versa,
    because bool is a subclass of int -- validate_command guards this explicitly).
    """

    name: str
    kind: ParamKind


@dataclass(slots=True, frozen=True)
class CommandSpec:
    """The dictionary entry for one command: target, schema, and hazard class.

    Args/fields:
        command_id: The opcode this spec describes.
        target: Canonical destination subsystem name (lowercase, e.g. "thermal"); iss_iface
            stamps CommandMsg.target from this, so the ground frame need not carry a target.
        params: The required parameters, in declaration order.
        hazardous: True if the command requires the ARM/EXECUTE two-step (Phase 6B). 6A
            commands are all non-hazardous.
    """

    command_id: CommandId
    target: str
    params: tuple[ParamSpec, ...]
    hazardous: bool


COMMAND_DICTIONARY: dict[CommandId, CommandSpec] = {
    CommandId.PING: CommandSpec(CommandId.PING, "iss_iface", (), hazardous=False),
    CommandId.NOOP: CommandSpec(CommandId.NOOP, "iss_iface", (), hazardous=False),
    CommandId.SET_THERMAL_LIMIT: CommandSpec(
        CommandId.SET_THERMAL_LIMIT,
        "thermal",
        (ParamSpec("limit_c", ParamKind.FLOAT),),
        hazardous=False,
    ),
}


def lookup_command(command_id: str) -> Result[CommandSpec, FaultCode]:
    """Resolve a wire command_id string to its CommandSpec.

    Args:
        command_id: The opcode string from the command body.

    Returns:
        Ok(spec) if command_id names a known CommandId, else Err(FaultCode.COMMAND_INVALID).
    """
    try:
        key = CommandId(command_id)
    except ValueError:
        return Err(FaultCode.COMMAND_INVALID)
    spec = COMMAND_DICTIONARY.get(key)
    if spec is None:
        return Err(FaultCode.COMMAND_INVALID)
    return Ok(spec)


def validate_command(
    spec: CommandSpec, params: dict[str, str | int | float | bool]
) -> Result[None, FaultCode]:
    """Check params against a spec's schema: exact key set + per-key primitive kind.

    Args:
        spec: The command spec to validate against.
        params: The parameter dict from the decoded command body.

    Returns:
        Ok(None) if params exactly matches the spec's declared parameters by name and kind,
        else Err(FaultCode.COMMAND_INVALID).

    Notes:
        bool is a subclass of int in Python; this rejects a bool where INT/FLOAT is required
        and rejects int/float where BOOL is required, so kinds are enforced strictly.
    """
    expected_names = {p.name for p in spec.params}
    if set(params.keys()) != expected_names:
        return Err(FaultCode.COMMAND_INVALID)
    for param in spec.params:
        value = params[param.name]
        if isinstance(value, bool) != (param.kind is ParamKind.BOOL):
            return Err(FaultCode.COMMAND_INVALID)
        if param.kind is ParamKind.FLOAT:
            if not isinstance(value, (int, float)):
                return Err(FaultCode.COMMAND_INVALID)
        elif not isinstance(value, _KIND_TO_TYPE[param.kind]):
            return Err(FaultCode.COMMAND_INVALID)
    return Ok(None)
```

`packages/flight/src/flight/libs/commands/__init__.py`:

```python
"""Typed command dictionary (see flight.libs.commands.dictionary)."""

from flight.libs.commands.dictionary import (
    COMMAND_DICTIONARY,
    CommandSpec,
    ParamSpec,
    lookup_command,
    validate_command,
)

__all__ = [
    "COMMAND_DICTIONARY",
    "CommandSpec",
    "ParamSpec",
    "lookup_command",
    "validate_command",
]
```

- [ ] **Step 4: Run -- expect PASS**

Run: `uv run pytest packages/flight/tests/test_command_dictionary.py -q` -> PASS.

- [ ] **Step 5: Gates + commit**

> NOTE: `flight.libs.commands` imports only `flight.libs.types` -- fine under `libs-layers` (which only orders `messages > types`). Confirm `uv run lint-imports` stays green.

```
git add packages/flight/src/flight/libs/commands/__init__.py packages/flight/src/flight/libs/commands/dictionary.py packages/flight/tests/test_command_dictionary.py
git commit -m "feat(libs): add typed command dictionary + param-schema validator"
```

---

## Task 5: Link + command-ingress config

**Files:**
- Modify: `packages/flight/src/flight/libs/config/config.py`
- Modify: `packages/flight/src/flight/libs/config/__init__.py`
- Modify: `config/default.toml`
- Modify: `packages/flight/src/flight/core/config_loader.py`
- Test: `packages/flight/tests/test_config_defaults.py`, and add range-validation tests (find the loader test, e.g. `packages/flight/tests/test_config_loader.py`)

- [ ] **Step 1: Add `LinkConfig` and `CommandIngressConfig` dataclasses**

In `config.py`, add (frozen, no slots, inline comments with units; follow the existing style):

```python
@dataclass(frozen=True)
class LinkConfig:
    """Station data-link transport config: CCSDS endpoints + APIDs.

    Commands arrive as CCSDS TC packets over a TCP server socket the payload binds; telemetry
    and products are sent as CCSDS TM packets over UDP to the station endpoint. Sockets open
    lazily in the real driver; SIL uses the byte-level sim link and ignores host/port.
    """

    command_tcp_host: str = "127.0.0.1"  # bind address for inbound TC server socket
    command_tcp_port: int = 50501  # TCP port the payload listens on for commands
    telemetry_udp_host: str = "127.0.0.1"  # station endpoint for outbound TM
    telemetry_udp_port: int = 50502  # UDP port for outbound telemetry/products
    socket_timeout_s: float = 1.0  # accept/recv timeout so the link thread can stop promptly
    tc_apid: int = 0x001  # CCSDS APID for inbound telecommands
    tm_apid: int = 0x002  # CCSDS APID for outbound telemetry


@dataclass(frozen=True)
class CommandIngressConfig:
    """Command-ingress integrity + authentication config.

    The HMAC key is loaded from hmac_key_path by the composition root and injected into
    iss_iface (not read by the app). sequence_window is the per-source replay guard: a command
    whose seq is <= the last accepted seq for that source is rejected as a replay/duplicate.
    """

    hmac_key_path: str = "data/keys/uplink_hmac.key"  # path to the shared HMAC secret
    require_auth: bool = True  # if False, skip HMAC verification (test/bench only)
    accepted_sources: tuple[str, ...] = ("ground", "station_ops")  # allowed command origins
```

Add the two fields to `PactConfig` (after `gimbal`, using `field(default_factory=...)`):

```python
    link: LinkConfig = field(default_factory=LinkConfig)
    command_ingress: CommandIngressConfig = field(default_factory=CommandIngressConfig)
```

- [ ] **Step 2: Re-export both classes** from `config/__init__.py` (import block + `__all__`, alphabetical).

- [ ] **Step 3: Add the TOML sections** to `config/default.toml` -- every field, value-identical to the dataclass defaults (the `test_config_defaults` guard requires it):

```toml
[link]
command_tcp_host = "127.0.0.1"
command_tcp_port = 50501
telemetry_udp_host = "127.0.0.1"
telemetry_udp_port = 50502
socket_timeout_s = 1.0
tc_apid = 1
tm_apid = 2

[command_ingress]
hmac_key_path = "data/keys/uplink_hmac.key"
require_auth = true
accepted_sources = ["ground", "station_ops"]
```

- [ ] **Step 4: Map the new sections in `config_loader._build_pact_config`**

Add the imports (`LinkConfig`, `CommandIngressConfig`) and two mapping blocks, mirroring the existing `.get(key, Default)` idiom, then pass them into the `PactConfig(...)` constructor:

```python
    link_sect = data.get("link", {})
    link = LinkConfig(
        command_tcp_host=str(link_sect.get("command_tcp_host", LinkConfig.command_tcp_host)),
        command_tcp_port=int(link_sect.get("command_tcp_port", LinkConfig.command_tcp_port)),
        telemetry_udp_host=str(
            link_sect.get("telemetry_udp_host", LinkConfig.telemetry_udp_host)
        ),
        telemetry_udp_port=int(
            link_sect.get("telemetry_udp_port", LinkConfig.telemetry_udp_port)
        ),
        socket_timeout_s=float(link_sect.get("socket_timeout_s", LinkConfig.socket_timeout_s)),
        tc_apid=int(link_sect.get("tc_apid", LinkConfig.tc_apid)),
        tm_apid=int(link_sect.get("tm_apid", LinkConfig.tm_apid)),
    )
    ingress_sect = data.get("command_ingress", {})
    command_ingress = CommandIngressConfig(
        hmac_key_path=str(
            ingress_sect.get("hmac_key_path", CommandIngressConfig.hmac_key_path)
        ),
        require_auth=bool(ingress_sect.get("require_auth", CommandIngressConfig.require_auth)),
        accepted_sources=tuple(
            str(s)
            for s in ingress_sect.get("accepted_sources", CommandIngressConfig.accepted_sources)
        ),
    )
```

> NOTE: `CommandIngressConfig.field` default access works because these are dataclass-level
> defaults on a frozen dataclass; use the same pattern the loader already uses for other
> sections. If the loader references `DataclassType.field` for defaults elsewhere, match it.

- [ ] **Step 5: Implement range validation in `config_loader._validate`**

Replace the `return None  # placeholder` stub body with real range checks for the safety-relevant new fields (and keep returning `None` when all pass). Validate within the existing `_validate(data)` signature, returning an error string on the first violation:

```python
    link = data.get("link", {})
    for apid_key in ("tc_apid", "tm_apid"):
        apid = link.get(apid_key, 0)
        if not (0 <= int(apid) <= 0x7FF):
            return f"link.{apid_key} must fit in 11 bits (0..0x7FF)"
    for port_key in ("command_tcp_port", "telemetry_udp_port"):
        port = int(link.get(port_key, 0))
        if not (1 <= port <= 65535):
            return f"link.{port_key} must be in 1..65535"
    if float(link.get("socket_timeout_s", 1.0)) <= 0.0:
        return "link.socket_timeout_s must be > 0"
    ingress = data.get("command_ingress", {})
    if bool(ingress.get("require_auth", True)) and not str(ingress.get("hmac_key_path", "")):
        return "command_ingress.hmac_key_path must be set when require_auth is true"
```

- [ ] **Step 6: Register sections in `test_config_defaults.py`**

Add `"link": LinkConfig` and `"command_ingress": CommandIngressConfig` to the `_SECTION_TO_DATACLASS` map (and its import block).

- [ ] **Step 7: Add loader range-validation tests**

In the loader test module, add tests that `load_config` returns `Err` for an out-of-range APID and an out-of-range port (write a tiny override dict/temp TOML and assert `isinstance(result, Err)`). Follow the existing loader-test structure.

- [ ] **Step 8: Run config tests + gates, commit**

Run: `uv run pytest packages/flight/tests/test_config_defaults.py packages/flight/tests/test_config_loader.py -q` then full gates.

```
git add packages/flight/src/flight/libs/config/config.py packages/flight/src/flight/libs/config/__init__.py config/default.toml packages/flight/src/flight/core/config_loader.py packages/flight/tests/test_config_defaults.py packages/flight/tests/test_config_loader.py
git commit -m "feat(config): add link transport + command-ingress config with range validation"
```

---

## Task 6: Byte-level `StationLink` Protocol + sim driver + real socket driver

This task changes the HAL contract and both drivers together, plus `main.py`, in one commit (the Protocol change must land atomically to keep gates green). `iss_iface` is NOT touched here -- it keeps using the *old* methods, which remain on the Protocol until Task 8 removes them. So during this task the Protocol carries BOTH the old (`receive_command`/`send_downlink`) and new (`receive_packet`/`send_packet`/`link_state`/`close`) methods (expand phase of expand/migrate/contract).

**Files:**
- Modify: `packages/flight/src/flight/hal/interfaces/station.py`
- Modify: `packages/flight/src/flight/hal/drivers_sim/station.py`
- Modify: `packages/flight/src/flight/hal/drivers_real/station.py`
- Modify: `packages/flight/src/flight/core/main.py`
- Test: `packages/flight/tests/test_real_station_link.py` (new), `test_hal_interfaces.py`, `test_real_drivers.py`

- [ ] **Step 1: Write failing loopback tests for the real driver**

`packages/flight/tests/test_real_station_link.py`:

```python
"""RealStationLink loopback-socket integration tests (real TCP/UDP, localhost)."""

import socket

from flight.hal.drivers_real import RealStationLink
from flight.libs.ccsds import CcsdsHeader, encode_packet, packet_length
from flight.libs.config import LinkConfig
from flight.libs.time import ManualClock
from flight.libs.types import LinkState, Ok


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_receive_packet_deframes_tcp_stream() -> None:
    cfg = LinkConfig(command_tcp_host="127.0.0.1", command_tcp_port=_free_port(),
                     telemetry_udp_host="127.0.0.1", telemetry_udp_port=_free_port(),
                     socket_timeout_s=0.5)
    link = RealStationLink(cfg=cfg, clock=ManualClock())
    try:
        # Before any client connects: LOS, no command.
        assert link.link_state() is LinkState.LOS
        assert isinstance(link.receive_command(), Ok)  # legacy method still present, Ok(None)
        client = socket.create_connection((cfg.command_tcp_host, cfg.command_tcp_port), timeout=2.0)
        pkt = encode_packet(CcsdsHeader(packet_type=1, apid=cfg.tc_apid, sequence_count=0), b"hi").value
        client.sendall(pkt + pkt)  # two packets back-to-back to prove deframing
        # poll until both packets surface
        seen = []
        for _ in range(50):
            result = link.receive_packet()
            assert isinstance(result, Ok)
            if result.value is not None:
                seen.append(result.value)
            if len(seen) == 2:
                break
        assert seen == [pkt, pkt]
        assert link.link_state() is LinkState.AOS
        client.close()
    finally:
        link.close()


def test_send_packet_emits_udp() -> None:
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(2.0)
    udp_port = int(rx.getsockname()[1])
    cfg = LinkConfig(command_tcp_host="127.0.0.1", command_tcp_port=_free_port(),
                     telemetry_udp_host="127.0.0.1", telemetry_udp_port=udp_port,
                     socket_timeout_s=0.5)
    link = RealStationLink(cfg=cfg, clock=ManualClock())
    try:
        payload = b"telemetry-bytes"
        assert isinstance(link.send_packet(payload), Ok)
        data, _ = rx.recvfrom(4096)
        assert data == payload
    finally:
        link.close()
        rx.close()
```

- [ ] **Step 2: Run -- expect failure**

Run: `uv run pytest packages/flight/tests/test_real_station_link.py -q`
Expected: FAIL (`RealStationLink` has no `cfg`/`clock` ctor, no `receive_packet`/`send_packet`/`link_state`/`close`).

- [ ] **Step 3: Extend the `StationLink` Protocol (byte-level methods alongside the old ones)**

Rewrite `hal/interfaces/station.py` to add the new methods (keep the two legacy methods for now), updating the module docstring to describe the byte-level transport. The Protocol body becomes:

```python
from typing import Protocol, runtime_checkable

from flight.libs.messages import CommandMsg, DownlinkItemMsg
from flight.libs.types import FaultCode, LinkState, Result


@runtime_checkable
class StationLink(Protocol):
    """Hardware abstraction for the ISS/station command + downlink interface.

    The link is a pure byte transport for CCSDS Space Packets: telecommands inbound,
    telemetry/products outbound. Framing, CRC, authentication, and validation live in
    iss_iface, not here. (The legacy command-level methods are retained transitionally and
    are removed once iss_iface migrates to the byte-level API.)
    """

    def receive_packet(self) -> Result[bytes | None, FaultCode]:
        """Pop the next complete inbound CCSDS packet, or Ok(None) if none is pending."""
        ...

    def send_packet(self, packet: bytes) -> Result[None, FaultCode]:
        """Transmit one complete outbound CCSDS packet (bytes) to the station."""
        ...

    def link_state(self) -> LinkState:
        """Return the current AOS/LOS acquisition state."""
        ...

    def close(self) -> None:
        """Release any sockets/threads. Safe to call multiple times."""
        ...

    def receive_command(self) -> Result[CommandMsg | None, FaultCode]:
        """Deprecated (removed once iss_iface migrates); legacy command-level uplink."""
        ...

    def send_downlink(self, item: DownlinkItemMsg) -> Result[None, FaultCode]:
        """Deprecated (removed once iss_iface migrates); legacy command-level downlink."""
        ...
```

- [ ] **Step 4: Extend `SimStationLink` with the byte-level API**

In `drivers_sim/station.py`, add a second inbound source for packets and a scriptable link state, while keeping the legacy methods. Replace the class with:

```python
"""Simulated station link (byte-level + legacy command-level during migration).

Replays scripted inbound CCSDS packets (one per receive_packet() call, then Ok(None)) and
records every outbound packet. Link state is scriptable (defaults AOS). Retains the legacy
receive_command/send_downlink during the iss_iface migration. Satisfies StationLink
structurally; used by SIL and tests.
"""

from flight.libs.messages import CommandMsg, DownlinkItemMsg
from flight.libs.types import FaultCode, LinkState, Ok, Result


class SimStationLink:
    """Station link replaying scripted packets and recording outbound packets (sim/SIL)."""

    def __init__(
        self, inbound: list[bytes] | None = None, link_state: LinkState = LinkState.AOS
    ) -> None:
        """Initialize with inbound packets to replay, in order, and a fixed link state.

        Args:
            inbound: CCSDS packets returned one per receive_packet() call, in order.
            link_state: The AOS/LOS state link_state() reports (default AOS).
        """
        self._inbound: list[bytes] = list(inbound) if inbound is not None else []
        self._index = 0
        self._sent: list[bytes] = []
        self._link_state = link_state

    def enqueue(self, packet: bytes) -> None:
        """Append an inbound packet to be returned by a later receive_packet() call."""
        self._inbound.append(packet)

    def set_link_state(self, state: LinkState) -> None:
        """Set the AOS/LOS state reported by link_state() (test/SIL hook)."""
        self._link_state = state

    def receive_packet(self) -> Result[bytes | None, FaultCode]:
        """Return the next scripted inbound packet, or Ok(None) once exhausted."""
        if self._index >= len(self._inbound):
            return Ok(None)
        packet = self._inbound[self._index]
        self._index += 1
        return Ok(packet)

    def send_packet(self, packet: bytes) -> Result[None, FaultCode]:
        """Record the outbound packet and return Ok(None)."""
        self._sent.append(packet)
        return Ok(None)

    def link_state(self) -> LinkState:
        """Return the scripted AOS/LOS state."""
        return self._link_state

    def close(self) -> None:
        """No-op for the sim link."""

    @property
    def sent(self) -> tuple[bytes, ...]:
        """All packets passed to send_packet, in order (test/SIL inspection hook)."""
        return tuple(self._sent)

    # --- legacy command-level API (removed in Task 8) ---
    def receive_command(self) -> Result[CommandMsg | None, FaultCode]:
        """Legacy no-op during migration: always Ok(None)."""
        return Ok(None)

    def send_downlink(self, item: DownlinkItemMsg) -> Result[None, FaultCode]:
        """Legacy no-op during migration: accept and drop."""
        return Ok(None)
```

- [ ] **Step 5: Implement the real socket driver**

Rewrite `drivers_real/station.py` as a real TCP-inbound / UDP-outbound CCSDS link. Sockets open lazily in `__init__`; a daemon thread accepts one client and recv-loops, deframing the byte stream into complete packets via `packet_length`, pushing each onto a `deque` guarded by a `Lock`. `receive_packet` pops non-blocking. `send_packet` `sendto`s UDP. `link_state` is AOS while a client is connected. `close` stops the thread and closes sockets. Raise `ValueError` on bad config.

```python
"""Real ISS/station data-link driver: CCSDS TC over TCP in, TM over UDP out.

Inbound telecommands arrive on a TCP server socket the payload binds (one station client);
a background daemon thread accepts the client, recv-loops, and deframes the byte stream into
complete CCSDS packets (using packet_length on the 6-byte header) which receive_packet() pops
non-blocking. Outbound telemetry/products are sent as UDP datagrams to the station endpoint.
Link state is AOS while a client is connected, LOS otherwise. Sockets are stdlib (no SDK) but
open lazily in __init__; close() tears down the thread and sockets. Library methods return
Result and never raise; only bad startup config raises ValueError.

Satisfies: REQ-COMM-HIGH-001, REQ-COMM-HIGH-002.
"""

from __future__ import annotations

# stdlib
import socket
import threading
from collections import deque

# internal
from flight.libs.ccsds import CCSDS_PRIMARY_HEADER_SIZE, packet_length
from flight.libs.config import LinkConfig
from flight.libs.messages import CommandMsg, DownlinkItemMsg
from flight.libs.time import Clock
from flight.libs.types import Err, FaultCode, LinkState, Ok, Result


class RealStationLink:
    """Real CCSDS station link (TCP command-in, UDP telemetry-out). Satisfies StationLink."""

    def __init__(self, cfg: LinkConfig, clock: Clock) -> None:
        """Bind the inbound TCP server and outbound UDP socket; start the accept/recv thread.

        Args:
            cfg: Link transport config (hosts/ports/APIDs/timeout).
            clock: Injected clock (reserved for future timestamping / LOS timeouts).

        Raises:
            ValueError: if host is empty or a port is out of range (startup misconfig).
        """
        if not cfg.command_tcp_host or not cfg.telemetry_udp_host:
            raise ValueError("RealStationLink requires non-empty hosts")
        if not (1 <= cfg.command_tcp_port <= 65535 and 1 <= cfg.telemetry_udp_port <= 65535):
            raise ValueError("RealStationLink requires ports in 1..65535")
        self._cfg = cfg
        self._clock = clock
        self._lock = threading.Lock()
        self._inbound: deque[bytes] = deque()
        self._connected = False
        self._stop = threading.Event()

        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((cfg.command_tcp_host, cfg.command_tcp_port))
        self._server.listen(1)
        self._server.settimeout(cfg.socket_timeout_s)
        self._udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self._thread = threading.Thread(target=self._serve, name="station_link", daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        """Accept one client and recv-loop, deframing packets onto the inbound deque."""
        while not self._stop.is_set():
            try:
                conn, _ = self._server.accept()
            except (TimeoutError, OSError):
                continue
            conn.settimeout(self._cfg.socket_timeout_s)
            with self._lock:
                self._connected = True
            self._recv_loop(conn)
            with self._lock:
                self._connected = False

    def _recv_loop(self, conn: socket.socket) -> None:
        """Read a TCP stream, splitting it into complete CCSDS packets by the length field."""
        buffer = bytearray()
        with conn:
            while not self._stop.is_set():
                try:
                    chunk = conn.recv(4096)
                except (TimeoutError, OSError):
                    continue
                if not chunk:
                    return  # peer closed
                buffer.extend(chunk)
                while len(buffer) >= CCSDS_PRIMARY_HEADER_SIZE:
                    length_result = packet_length(bytes(buffer[:CCSDS_PRIMARY_HEADER_SIZE]))
                    if isinstance(length_result, Err):
                        buffer.clear()
                        break
                    total = length_result.value
                    if len(buffer) < total:
                        break
                    with self._lock:
                        self._inbound.append(bytes(buffer[:total]))
                    del buffer[:total]

    def receive_packet(self) -> Result[bytes | None, FaultCode]:
        """Pop the next deframed inbound packet, or Ok(None) if none is pending."""
        with self._lock:
            if self._inbound:
                return Ok(self._inbound.popleft())
        return Ok(None)

    def send_packet(self, packet: bytes) -> Result[None, FaultCode]:
        """Send one packet as a UDP datagram to the station telemetry endpoint."""
        try:
            self._udp.sendto(packet, (self._cfg.telemetry_udp_host, self._cfg.telemetry_udp_port))
        except OSError:
            return Err(FaultCode.COMM_TIMEOUT)
        return Ok(None)

    def link_state(self) -> LinkState:
        """Return AOS while a station client is connected, LOS otherwise."""
        with self._lock:
            return LinkState.AOS if self._connected else LinkState.LOS

    def close(self) -> None:
        """Stop the accept/recv thread and close both sockets (idempotent)."""
        self._stop.set()
        try:
            self._server.close()
        except OSError:
            pass
        try:
            self._udp.close()
        except OSError:
            pass
        self._thread.join(timeout=2.0)

    # --- legacy command-level API (removed in Task 8) ---
    def receive_command(self) -> Result[CommandMsg | None, FaultCode]:
        """Legacy no-op during migration: always Ok(None)."""
        return Ok(None)

    def send_downlink(self, item: DownlinkItemMsg) -> Result[None, FaultCode]:
        """Legacy no-op during migration: accept and drop."""
        return Ok(None)
```

> NOTE: export `CCSDS_PRIMARY_HEADER_SIZE` from `flight.libs.ccsds.__init__` (add it to the import block + `__all__`) so the real driver can import it.

- [ ] **Step 6: Update `main.py` to construct the real link with config + clock**

In `core/main.py` `build_flight_system`, change `station=RealStationLink()` to `station=RealStationLink(cfg=config.link, clock=clock)`. Update the surrounding docstring note about the link (sockets bind lazily; ValueError on bad host/port). The `clock` is already in scope in `build_flight_system`; if not, thread it from `main()` (it constructs `RealClock()`).

- [ ] **Step 7: Update HAL interface tests**

`test_hal_interfaces.py`: the `test_real_station_link_satisfies_station_link` (or equivalent) must now construct `RealStationLink(cfg=LinkConfig(...free ports...), clock=ManualClock())` and `link.close()` after. `SimStationLink()` now takes no required args. Assert `isinstance(link, StationLink)`.

`test_real_drivers.py`: if it constructs `RealStationLink()` bare, update to pass `cfg`/`clock` (use free ports) and `close()`; sockets are stdlib so no SDK-absent skip is needed for the link.

- [ ] **Step 8: Run link tests + gates, commit**

Run: `uv run pytest packages/flight/tests/test_real_station_link.py packages/flight/tests/test_hal_interfaces.py packages/flight/tests/test_real_drivers.py -q` then full gates.

```
git add packages/flight/src/flight/hal/interfaces/station.py packages/flight/src/flight/hal/drivers_sim/station.py packages/flight/src/flight/hal/drivers_real/station.py packages/flight/src/flight/libs/ccsds/__init__.py packages/flight/src/flight/core/main.py packages/flight/tests/test_real_station_link.py packages/flight/tests/test_hal_interfaces.py packages/flight/tests/test_real_drivers.py
git commit -m "feat(hal): byte-level StationLink + real CCSDS TCP/UDP socket driver"
```

---

## Task 7: `iss_iface` command-ingress pure core + front-door rewrite

**Files:**
- Create: `packages/flight/src/flight/iss_iface/ingress/__init__.py`
- Create: `packages/flight/src/flight/iss_iface/ingress/pipeline.py`
- Modify: `packages/flight/src/flight/iss_iface/app.py`
- Modify: `packages/flight/src/flight/core/composition.py`
- Modify: `packages/flight/src/flight/core/main.py`
- Modify: `packages/sim/src/sim/sil/runner.py` (only the `uplink_key` plumbing through `build_apps`; the bytes switch is Task 8)
- Test: `packages/flight/tests/test_iss_ingress_pipeline.py` (new), `test_iss_iface_app.py` (rewrite)

- [ ] **Step 1: Write failing ingress-pipeline tests**

`packages/flight/tests/test_iss_ingress_pipeline.py`:

```python
"""Pure command-ingress pipeline tests: decode -> CRC -> auth -> parse -> validate -> dedup."""

import hashlib
import hmac
import json

from flight.iss_iface.ingress import build_tc_packet, process_inbound
from flight.libs.types import AckStatus, FaultCode

_KEY = b"unit-test-key-0000000000000000000"


def _packet(command_id: str, params: dict, source: str, seq: int, key: bytes = _KEY) -> bytes:
    return build_tc_packet(command_id, params, source, seq, key, apid=1)


def test_accepts_valid_signed_command() -> None:
    outcome, last_seq = process_inbound(
        _packet("SET_THERMAL_LIMIT", {"limit_c": 70.0}, "ground", 1),
        key=_KEY, require_auth=True, accepted_sources=("ground",), last_seq={},
    )
    assert outcome.status is AckStatus.ACCEPTED
    assert outcome.command is not None
    assert outcome.command.target == "thermal"  # stamped from the dictionary
    assert last_seq["ground"] == 1


def test_rejects_crc_corruption() -> None:
    pkt = bytearray(_packet("PING", {}, "ground", 1))
    pkt[7] ^= 0xFF
    outcome, _ = process_inbound(bytes(pkt), key=_KEY, require_auth=True,
                                 accepted_sources=("ground",), last_seq={})
    assert outcome.status is AckStatus.REJECTED
    assert outcome.fault_code is FaultCode.COMMAND_CRC_FAIL


def test_rejects_bad_hmac() -> None:
    outcome, _ = process_inbound(_packet("PING", {}, "ground", 1, key=b"wrong-key"),
                                 key=_KEY, require_auth=True,
                                 accepted_sources=("ground",), last_seq={})
    assert outcome.status is AckStatus.REJECTED
    assert outcome.fault_code is FaultCode.COMMAND_AUTH_FAIL


def test_rejects_unknown_command() -> None:
    outcome, _ = process_inbound(_packet("LAUNCH_NUKE", {}, "ground", 1),
                                 key=_KEY, require_auth=True,
                                 accepted_sources=("ground",), last_seq={})
    assert outcome.status is AckStatus.REJECTED
    assert outcome.fault_code is FaultCode.COMMAND_INVALID


def test_rejects_replay() -> None:
    outcome, _ = process_inbound(_packet("PING", {}, "ground", 5),
                                 key=_KEY, require_auth=True,
                                 accepted_sources=("ground",), last_seq={"ground": 5})
    assert outcome.status is AckStatus.REJECTED
    assert outcome.fault_code is FaultCode.COMMAND_SEQ_ERROR


def test_rejects_unaccepted_source() -> None:
    outcome, _ = process_inbound(_packet("PING", {}, "intruder", 1),
                                 key=_KEY, require_auth=True,
                                 accepted_sources=("ground",), last_seq={})
    assert outcome.status is AckStatus.REJECTED
    assert outcome.fault_code is FaultCode.COMMAND_AUTH_FAIL
```

- [ ] **Step 2: Run -- expect ImportError**

Run: `uv run pytest packages/flight/tests/test_iss_ingress_pipeline.py -q` -> FAIL.

- [ ] **Step 3: Implement the ingress pipeline (pure)**

`packages/flight/src/flight/iss_iface/ingress/pipeline.py`:

```python
"""Pure command-ingress pipeline for iss_iface: bytes -> validated CommandMsg or rejection.

Stages, in order: CCSDS decode + CRC (flight.libs.ccsds) -> HMAC-SHA256 authentication over
the command body -> JSON parse -> source allow-list -> typed dictionary validation
(flight.libs.commands) -> monotonic per-source sequence (replay guard). The functions are
pure: no bus, no clock, no I/O, no logging. The app shell owns the HMAC key, the per-source
last-seq map, the clock, and the bus, and turns an IngressOutcome into a CommandMsg + an
always-emitted CommandAckMsg.

Wire format (TC): [CCSDS header type=1] [body = JSON {command_id, params, source, seq}]
[HMAC-SHA256 tag, 32 bytes] [CRC-32 trailer]. The dictionary stamps the canonical target;
the ground frame does not carry target.

Contains:
  - IngressOutcome: the per-packet result (command-or-None + ack status + reason + echo).
  - build_tc_packet: construct a signed TC packet (used by GSE/sim/tests, not flight).
  - process_inbound: run the full pipeline for one raw packet (Result-free; outcome-typed).

Satisfies: REQ-COMM-HIGH-003, REQ-COMM-HIGH-004.
"""

from __future__ import annotations

# stdlib
import hashlib
import hmac
import json
from dataclasses import dataclass

# internal
from flight.libs.ccsds import CcsdsHeader, decode_packet, encode_packet
from flight.libs.commands import lookup_command, validate_command
from flight.libs.messages import CommandMsg
from flight.libs.types import AckStatus, Err, FaultCode, MessageType

_HMAC_TAG_SIZE = 32  # SHA-256 digest length


@dataclass(slots=True)
class IngressOutcome:
    """Result of running one inbound packet through the ingress pipeline.

    Fields:
        command: The validated CommandMsg to publish, or None if rejected.
        status: ACCEPTED or REJECTED.
        fault_code: NONE on accept; the reject reason otherwise.
        command_id: Echoed opcode string ("" if the body was unparseable).
        source: Echoed origin ("" if unparseable).
        seq: Echoed sequence number (-1 if unparseable).
        detail: Human-readable context for the ack/fault.
    """

    command: CommandMsg | None
    status: AckStatus
    fault_code: FaultCode
    command_id: str
    source: str
    seq: int
    detail: str


def _reject(code: FaultCode, detail: str, command_id: str, source: str, seq: int) -> IngressOutcome:
    return IngressOutcome(None, AckStatus.REJECTED, code, command_id, source, seq, detail)


def build_tc_packet(
    command_id: str,
    params: dict[str, str | int | float | bool],
    source: str,
    seq: int,
    key: bytes,
    apid: int,
) -> bytes:
    """Construct a signed CCSDS telecommand packet (for GSE / sim / tests; not used in flight).

    Args:
        command_id / params / source / seq: The command body fields.
        key: The shared HMAC-SHA256 secret.
        apid: The telecommand APID.

    Returns:
        The framed TC packet bytes (header + body + HMAC tag + CRC trailer).

    Notes:
        params is JSON-serialized with sorted keys so the signed bytes are deterministic.
    """
    body = json.dumps(
        {"command_id": command_id, "params": params, "source": source, "seq": seq},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    tag = hmac.new(key, body, hashlib.sha256).digest()
    encoded = encode_packet(CcsdsHeader(packet_type=1, apid=apid, sequence_count=seq & 0x3FFF),
                            body + tag)
    if isinstance(encoded, Err):
        raise ValueError(f"could not encode TC packet: {encoded.error}")  # test helper only
    return encoded.value


def process_inbound(
    raw: bytes,
    key: bytes,
    require_auth: bool,
    accepted_sources: tuple[str, ...],
    last_seq: dict[str, int],
) -> tuple[IngressOutcome, dict[str, int]]:
    """Run one raw inbound packet through the full ingress pipeline.

    Args:
        raw: The complete framed CCSDS TC packet.
        key: The shared HMAC-SHA256 secret.
        require_auth: If False, skip the HMAC check (bench/test only).
        accepted_sources: The allow-list of command origins.
        last_seq: Per-source last-accepted sequence map (threaded state, not mutated here).

    Returns:
        (outcome, new_last_seq). On ACCEPTED, new_last_seq[source] is updated to the command's
        seq; on REJECTED, last_seq is returned unchanged. No exceptions: malformed input maps
        to a REJECTED outcome with the appropriate FaultCode.
    """
    decoded = decode_packet(raw)
    if isinstance(decoded, Err):
        return _reject(decoded.error, "ccsds decode/crc failed", "", "", -1), last_seq
    _header, data = decoded.value

    if len(data) < _HMAC_TAG_SIZE:
        return _reject(FaultCode.COMMAND_AUTH_FAIL, "missing hmac tag", "", "", -1), last_seq
    body, tag = data[:-_HMAC_TAG_SIZE], data[-_HMAC_TAG_SIZE:]

    try:
        fields = json.loads(body.decode("utf-8"))
        command_id = str(fields["command_id"])
        params = dict(fields["params"])
        source = str(fields["source"])
        seq = int(fields["seq"])
    except (ValueError, KeyError, TypeError):
        return _reject(FaultCode.COMMAND_INVALID, "malformed command body", "", "", -1), last_seq

    if require_auth:
        expected = hmac.new(key, body, hashlib.sha256).digest()
        if not hmac.compare_digest(expected, tag):
            return _reject(FaultCode.COMMAND_AUTH_FAIL, "hmac mismatch", command_id, source, seq), last_seq
    if source not in accepted_sources:
        return _reject(FaultCode.COMMAND_AUTH_FAIL, "source not accepted", command_id, source, seq), last_seq

    spec_result = lookup_command(command_id)
    if isinstance(spec_result, Err):
        return _reject(spec_result.error, "unknown command", command_id, source, seq), last_seq
    spec = spec_result.value
    valid = validate_command(spec, params)
    if isinstance(valid, Err):
        return _reject(valid.error, "param validation failed", command_id, source, seq), last_seq

    if seq <= last_seq.get(source, -1):
        return _reject(FaultCode.COMMAND_SEQ_ERROR, "replay/duplicate seq", command_id, source, seq), last_seq

    command = CommandMsg(
        msg_type=MessageType.COMMAND,
        timestamp_utc="",  # the shell stamps this with the clock
        target=spec.target,
        command_id=command_id,
        params=params,
        source=source,
        seq=seq,
    )
    new_last_seq = dict(last_seq)
    new_last_seq[source] = seq
    outcome = IngressOutcome(command, AckStatus.ACCEPTED, FaultCode.NONE, command_id, source, seq, "")
    return outcome, new_last_seq
```

`packages/flight/src/flight/iss_iface/ingress/__init__.py`:

```python
"""Command-ingress pipeline (see flight.iss_iface.ingress.pipeline)."""

from flight.iss_iface.ingress.pipeline import IngressOutcome, build_tc_packet, process_inbound

__all__ = ["IngressOutcome", "build_tc_packet", "process_inbound"]
```

- [ ] **Step 4: Run ingress tests -- expect PASS**

Run: `uv run pytest packages/flight/tests/test_iss_ingress_pipeline.py -q` -> PASS.

- [ ] **Step 5: Rewrite the `iss_iface` app shell**

Rewrite `iss_iface/app.py`. The app now: holds the injected `uplink_key`, the `LinkConfig`/`CommandIngressConfig`, a mutable `IngressState` (per-source last-seq + outbound TM seq), and subscriptions to `DownlinkItemMsg` and `CommandAckMsg`. `pump_uplink` drains `receive_packet`, runs `process_inbound`, publishes `CommandMsg` (stamped) on ACCEPTED and always publishes a `CommandAckMsg` (+ `FaultEventMsg` on REJECTED). `pump_downlink` only drains when AOS, encoding `CommandAckMsg`/`DownlinkItemMsg` into TM packets via `encode_packet` and sending via `send_packet`. `tick` also publishes a `LinkStateMsg`.

Key structure (full module; replace the file):

```python
"""ISS interface app: the authenticated command-ingress front door + downlink egress.

Inbound: receive_packet (raw CCSDS bytes) -> process_inbound (decode/CRC/HMAC/parse/validate/
dedup) -> publish CommandMsg (validated) + always publish a CommandAckMsg (ACCEPTED/REJECTED).
Outbound: when AOS, drain CommandAckMsg + DownlinkItemMsg, encode each into a CCSDS TM packet,
and send_packet. Each tick also publishes the current LinkStateMsg. The ingress decision logic
is a pure core (flight.iss_iface.ingress); this shell owns the bus, clock, HMAC key, and the
mutable sequence state.

Contains:
  - IngressState: mutable per-source last-seq map + outbound TM sequence counter.
  - IssIfaceApp: from_config(); pump_uplink(); pump_downlink(); tick(); run(); helpers.

Satisfies: REQ-COMM-HIGH-001, REQ-COMM-HIGH-003, REQ-COMM-HIGH-004.
"""

from __future__ import annotations

# stdlib
import json
import threading
from dataclasses import dataclass, field

# internal
from flight.hal.interfaces import StationLink
from flight.iss_iface.ingress import process_inbound
from flight.libs.bus import MessageBus, Subscription
from flight.libs.ccsds import CcsdsHeader, encode_packet
from flight.libs.config import CommandIngressConfig, FaultConfig, LinkConfig, PactConfig
from flight.libs.messages import (
    CommandAckMsg,
    CommandMsg,
    DownlinkItemMsg,
    FaultEventMsg,
    HeartbeatMsg,
    LinkStateMsg,
)
from flight.libs.time import Clock
from flight.libs.types import AckStatus, Err, FaultCode, LinkState, MessageType, Ok

HEARTBEAT_SUBSYSTEM = "iss_iface"


@dataclass(slots=True)
class IngressState:
    """Mutable ingress state owned by the app shell (threaded through the pure core)."""

    last_seq: dict[str, int] = field(default_factory=dict)
    tm_sequence: int = 0


@dataclass(frozen=True)
class IssIfaceApp:
    """Station <-> bus command-ingress front door + downlink egress."""

    fault_cfg: FaultConfig
    link_cfg: LinkConfig
    ingress_cfg: CommandIngressConfig
    uplink_key: bytes
    link: StationLink
    bus: MessageBus
    clock: Clock
    downlink: Subscription[DownlinkItemMsg]
    acks: Subscription[CommandAckMsg]
    state: IngressState

    @staticmethod
    def from_config(
        cfg: PactConfig,
        bus: MessageBus,
        clock: Clock,
        link: StationLink,
        uplink_key: bytes,
    ) -> IssIfaceApp:
        """Assemble an IssIfaceApp, subscribing to outbound downlink items and command acks.

        Args:
            cfg: Top-level PactConfig (fault for timing; link + command_ingress for ingress).
            bus / clock / link: Shared services and the injected station driver.
            uplink_key: The HMAC secret loaded by the composition root.

        Returns:
            An IssIfaceApp with fresh DownlinkItemMsg + CommandAckMsg subscriptions and empty
            ingress state.
        """
        return IssIfaceApp(
            fault_cfg=cfg.fault,
            link_cfg=cfg.link,
            ingress_cfg=cfg.command_ingress,
            uplink_key=uplink_key,
            link=link,
            bus=bus,
            clock=clock,
            downlink=bus.subscribe(DownlinkItemMsg),
            acks=bus.subscribe(CommandAckMsg),
            state=IngressState(),
        )

    def pump_uplink(self) -> int:
        """Drain inbound packets; publish validated CommandMsgs; always ack each packet.

        Returns:
            The number of CommandMsg published (accepted commands). Each inbound packet --
            accepted or rejected -- produces exactly one CommandAckMsg; rejects also emit a
            FaultEventMsg. A link Err stops the drain early (preserves ordering).
        """
        published = 0
        while True:
            result = self.link.receive_packet()
            if isinstance(result, Err):
                self._publish_fault(result.error, "station uplink receive failed")
                break
            raw = result.value
            if raw is None:
                break
            outcome, self.state.last_seq = process_inbound(
                raw,
                self.uplink_key,
                self.ingress_cfg.require_auth,
                self.ingress_cfg.accepted_sources,
                self.state.last_seq,
            )
            if outcome.command is not None:
                from dataclasses import replace

                self.bus.publish(replace(outcome.command, timestamp_utc=self.clock.wall_clock_iso()))
                published += 1
            else:
                self._publish_fault(outcome.fault_code, outcome.detail)
            self._publish_ack(outcome)
        return published

    def pump_downlink(self) -> int:
        """When AOS, encode and send pending command acks and downlink items as TM packets.

        Returns:
            The number of packets sent. During LOS nothing is drained (acks/items wait in the
            subscription queue). A send Err emits a fault and is not counted.
        """
        if self.link.link_state() is not LinkState.AOS:
            return 0
        sent = 0
        while not self.acks.empty():
            ack = self.acks.get_nowait()
            sent += self._send_tm(self._ack_to_json(ack))
        while not self.downlink.empty():
            item = self.downlink.get_nowait()
            sent += self._send_tm(item.payload_bytes)
        return sent

    def tick(self) -> None:
        """Publish link state, pump inbound commands, then pump outbound downlinks once."""
        self.bus.publish(
            LinkStateMsg(
                msg_type=MessageType.LINK_STATE,
                timestamp_utc=self.clock.wall_clock_iso(),
                state=self.link.link_state(),
            )
        )
        self.pump_uplink()
        self.pump_downlink()

    def run(self, stop_event: threading.Event) -> None:
        """Run the ingress loop until stop_event is set, emitting periodic heartbeats."""
        sequence = 0
        last_heartbeat = self.clock.monotonic_s()
        while not stop_event.is_set():
            self.tick()
            now = self.clock.monotonic_s()
            if now - last_heartbeat >= self.fault_cfg.watchdog_interval_s:
                self.bus.publish(
                    HeartbeatMsg(
                        msg_type=MessageType.HEARTBEAT,
                        timestamp_utc=self.clock.wall_clock_iso(),
                        subsystem=HEARTBEAT_SUBSYSTEM,
                        sequence=sequence,
                    )
                )
                sequence += 1
                last_heartbeat = now
            stop_event.wait(timeout=self.fault_cfg.watchdog_interval_s)
        self.link.close()

    def _send_tm(self, body: bytes) -> int:
        """Encode body into a CCSDS TM packet and send it; return 1 on success, 0 on error."""
        if len(body) == 0:
            return 0
        encoded = encode_packet(
            CcsdsHeader(packet_type=0, apid=self.link_cfg.tm_apid,
                        sequence_count=self.state.tm_sequence & 0x3FFF),
            body,
        )
        if isinstance(encoded, Err):
            self._publish_fault(encoded.error, "tm encode failed")
            return 0
        self.state.tm_sequence += 1
        result = self.link.send_packet(encoded.value)
        if isinstance(result, Ok):
            return 1
        self._publish_fault(result.error, "station downlink send failed")
        return 0

    def _ack_to_json(self, ack: CommandAckMsg) -> bytes:
        """Serialize a CommandAckMsg to compact JSON bytes for downlink."""
        return json.dumps(
            {
                "type": "command_ack",
                "status": ack.status.value,
                "command_id": ack.command_id,
                "source": ack.source,
                "seq": ack.seq,
                "fault_code": ack.fault_code.value,
                "detail": ack.detail,
            },
            separators=(",", ":"),
        ).encode("utf-8")

    def _publish_ack(self, outcome: "object") -> None:
        """Publish a CommandAckMsg for one ingress outcome (always, accept or reject)."""
        from flight.iss_iface.ingress import IngressOutcome

        assert isinstance(outcome, IngressOutcome)
        self.bus.publish(
            CommandAckMsg(
                msg_type=MessageType.COMMAND_ACK,
                timestamp_utc=self.clock.wall_clock_iso(),
                status=outcome.status,
                command_id=outcome.command_id,
                source=outcome.source,
                seq=outcome.seq,
                fault_code=outcome.fault_code,
                detail=outcome.detail,
            )
        )

    def _publish_fault(self, code: FaultCode, detail: str) -> None:
        """Publish a FaultEventMsg from the iss_iface subsystem onto the bus."""
        self.bus.publish(
            FaultEventMsg(
                msg_type=MessageType.FAULT_EVENT,
                timestamp_utc=self.clock.wall_clock_iso(),
                fault_code=code,
                subsystem=HEARTBEAT_SUBSYSTEM,
                detail=detail,
            )
        )
```

> NOTE: move the `from dataclasses import replace` and the `IngressOutcome` import to the module top (the inline imports above are shown for locality; final code groups imports at the top per conventions). Type `_publish_ack(self, outcome: IngressOutcome)` properly once imported.

- [ ] **Step 6: Thread `uplink_key` through `build_apps`**

In `core/composition.py`, add `uplink_key: bytes` as the last parameter of `build_apps`, and pass it to `IssIfaceApp.from_config(config, bus, clock, drivers.station, uplink_key)`. Update the `build_apps` docstring.

- [ ] **Step 7: Load + inject the key in the flight composition root**

In `core/main.py` `build_flight_system`, load the key bytes from `config.command_ingress.hmac_key_path` (read the file in binary; on missing file raise `SystemExit`/`RuntimeError` with a clear message -- a missing uplink key is a startup misconfig) and pass it to `build_apps(..., calib, uplink_key)`. Add a small helper `_load_uplink_key(path: str) -> bytes`.

- [ ] **Step 8: Thread the key through the SIL root (signature only; bytes switch is Task 8)**

In `sim/sil/runner.py` `build_sil_system`, add a parameter `uplink_key: bytes = b"sil-test-key-0000000000000000000"` and pass it as the final arg to `build_apps(...)`. (The `inbound_commands -> inbound_packets` change happens in Task 8; here only the `build_apps` call gains `uplink_key` so the tree compiles.)

- [ ] **Step 9: Rewrite `test_iss_iface_app.py`**

The old assertions (pump_uplink republishes CommandMsg verbatim) no longer hold. Rewrite to build a key + signed packets with `build_tc_packet`, construct `IssIfaceApp.from_config(PactConfig(...), bus, ManualClock(), SimStationLink([pkt]), key)`, `subscribe(CommandMsg)` + `subscribe(CommandAckMsg)`, `tick()`, and assert: a valid signed command yields one `CommandMsg` (with `target` stamped) + one `CommandAckMsg(ACCEPTED)`; a tampered packet yields zero `CommandMsg` + one `CommandAckMsg(REJECTED, COMMAND_AUTH_FAIL)`; downlink only drains when the sim link is AOS (`SimStationLink([], link_state=LinkState.LOS)` holds acks). Use the same `_KEY` for the app and the packet builder.

- [ ] **Step 10: Run iss_iface tests + full gates, commit**

Run: `uv run pytest packages/flight/tests/test_iss_ingress_pipeline.py packages/flight/tests/test_iss_iface_app.py -q` then all five gates.

```
git add packages/flight/src/flight/iss_iface/ingress/__init__.py packages/flight/src/flight/iss_iface/ingress/pipeline.py packages/flight/src/flight/iss_iface/app.py packages/flight/src/flight/core/composition.py packages/flight/src/flight/core/main.py packages/sim/src/sim/sil/runner.py packages/flight/tests/test_iss_ingress_pipeline.py packages/flight/tests/test_iss_iface_app.py
git commit -m "feat(iss_iface): authenticated CCSDS command ingress with ack/nack contract"
```

---

## Task 8: Contract cleanup -- remove legacy command-level link API; switch SIL to packets

**Files:**
- Modify: `packages/flight/src/flight/hal/interfaces/station.py`
- Modify: `packages/flight/src/flight/hal/drivers_sim/station.py`
- Modify: `packages/flight/src/flight/hal/drivers_real/station.py`
- Modify: `packages/sim/src/sim/sil/runner.py`
- Test: `packages/sim/tests/test_sil_closed_loop.py`, `test_hal_interfaces.py`

- [ ] **Step 1: Remove the legacy methods from the Protocol and both drivers**

Delete `receive_command` and `send_downlink` from `StationLink` (interfaces/station.py), `SimStationLink`, and `RealStationLink`. Remove now-unused imports (`CommandMsg`, `DownlinkItemMsg`) where they were only used by those methods. Update each module docstring to drop the "legacy/migration" wording.

- [ ] **Step 2: Switch `build_sil_system` to inbound packets**

In `sim/sil/runner.py`, rename the parameter `inbound_commands: list[CommandMsg]` to `inbound_packets: list[bytes]` and construct `SimStationLink(inbound_packets)`. Update the docstring. Remove the now-unused `CommandMsg` import if nothing else uses it.

- [ ] **Step 3: Update existing SIL tests**

In `packages/sim/tests/test_sil_closed_loop.py`, every `build_sil_system(..., inbound_commands=[], ...)` becomes `inbound_packets=[]`. (No test currently scripts commands, so `[]` is the only change; the new command-path SIL test is added in Task 9.)

- [ ] **Step 4: Update `test_hal_interfaces.py`**

Remove any assertion that exercised `receive_command`/`send_downlink`; keep the structural `isinstance(link, StationLink)` checks (now satisfied by the byte-level methods).

- [ ] **Step 5: Run gates, commit**

Run all five gates.

```
git add packages/flight/src/flight/hal/interfaces/station.py packages/flight/src/flight/hal/drivers_sim/station.py packages/flight/src/flight/hal/drivers_real/station.py packages/sim/src/sim/sil/runner.py packages/sim/tests/test_sil_closed_loop.py packages/flight/tests/test_hal_interfaces.py
git commit -m "refactor(hal): drop legacy command-level StationLink API; SIL uses CCSDS packets"
```

---

## Task 9: End-to-end SIL command-ingress test

**Files:**
- Test: `packages/sim/tests/test_sil_closed_loop.py` (add command-path tests)

- [ ] **Step 1: Write the SIL command-ingress tests**

Add tests that drive a signed command packet through the real flight apps over the sim link. Use the SIL test key default (`build_sil_system`'s `uplink_key` default) and `build_tc_packet` with the same key.

```python
def test_valid_command_flows_through_to_bus_and_acks() -> None:
    """A signed SET_THERMAL_LIMIT packet becomes a CommandMsg + an ACCEPTED ack in SIL."""
    key = b"sil-test-key-0000000000000000000"
    pkt = build_tc_packet("SET_THERMAL_LIMIT", {"limit_c": 70.0}, "ground", 1, key, apid=1)
    system = build_sil_system(
        PactConfig(), ManualClock(), build_frames(2), plume_detector(),
        inbound_packets=[pkt], thermal_readings=[20.0, 20.0], power_readings=[10.0, 10.0],
    )
    commands = system.bus.subscribe(CommandMsg)
    acks = system.bus.subscribe(CommandAckMsg)
    SilHarness(system).run_steps(2)
    routed = [c for c in _drain(commands) if c.command_id == "SET_THERMAL_LIMIT"]
    assert len(routed) == 1
    assert routed[0].target == "thermal"
    assert any(a.status is AckStatus.ACCEPTED for a in _drain(acks))


def test_tampered_command_is_rejected_not_routed() -> None:
    """A packet signed with the wrong key yields a REJECTED ack and no CommandMsg."""
    pkt = build_tc_packet("PING", {}, "ground", 1, b"wrong-key-xxxxxxxxxxxxxxxxxxxxxxx", apid=1)
    system = build_sil_system(
        PactConfig(), ManualClock(), build_frames(2), plume_detector(),
        inbound_packets=[pkt], thermal_readings=[20.0, 20.0], power_readings=[10.0, 10.0],
    )
    commands = system.bus.subscribe(CommandMsg)
    acks = system.bus.subscribe(CommandAckMsg)
    SilHarness(system).run_steps(2)
    assert not [c for c in _drain(commands) if c.source == "ground"]
    rejects = [a for a in _drain(acks) if a.status is AckStatus.REJECTED]
    assert rejects and rejects[0].fault_code is FaultCode.COMMAND_AUTH_FAIL
```

Add a small `_drain(subscription)` helper (or reuse the existing test pattern) and the imports (`build_tc_packet` from `flight.iss_iface.ingress`, `CommandAckMsg`/`CommandMsg` from `flight.libs.messages`, `AckStatus`/`FaultCode` from `flight.libs.types`).

> NOTE: confirm the SIL test key string is exactly the `build_sil_system` default. If you prefer, pass `uplink_key=key` explicitly to `build_sil_system` in the test to avoid coupling to the default.

- [ ] **Step 2: Run -- expect PASS**

Run: `uv run pytest packages/sim/tests/test_sil_closed_loop.py -q` -> PASS.

- [ ] **Step 3: Gates + commit**

```
git add packages/sim/tests/test_sil_closed_loop.py
git commit -m "test(sil): prove authenticated command ingress end-to-end through the flight apps"
```

---

## Task 10: Documentation -- CONTEXT files, ADR, ADR index

**Files:**
- Modify: `packages/flight/src/flight/iss_iface/CONTEXT.md`
- Modify: `packages/flight/src/flight/hal/CONTEXT.md`
- Modify: `packages/flight/src/flight/libs/CONTEXT.md`
- Modify: `packages/sim/src/sim/CONTEXT.md`
- Create: `docs/adr/0009-iss-link-transport-command-ingress.md`
- Modify: `docs/adr/README.md`

- [ ] **Step 1: Rewrite `iss_iface/CONTEXT.md`**

Reverse the "zero command interpretation / pure transport bridge" framing. Document the new reality: iss_iface is the authenticated command-ingress front door (decode -> CRC -> HMAC -> dictionary validate -> dedup -> CommandMsg + always-ack) and the downlink egress (encode acks/items to CCSDS TM, AOS-gated). Note: the link is now a byte transport; the ingress decision logic is a pure core (`ingress/pipeline.py`); the shell owns the HMAC key (injected) and the mutable per-source seq + outbound TM seq; new fault codes are log-and-continue (NACK, never SAFE); the command router + hazardous ARM/EXECUTE is Phase 6B (not here). Update the "Explicitly Out of Scope" section (CCSDS framing is now IN scope; model-chunk reassembly is still out).

- [ ] **Step 2: Update `hal/CONTEXT.md`** (station section): `StationLink` is byte-level (`receive_packet`/`send_packet`/`link_state`/`close`); `RealStationLink` is a real TCP-in/UDP-out CCSDS link (lazy sockets, daemon accept/recv thread, `packet_length` deframing, AOS=connected, `ValueError` on bad config); `SimStationLink` replays/records packets with scriptable link state; drivers may import `flight.libs.ccsds`.

- [ ] **Step 3: Update `libs/CONTEXT.md`**: new `flight.libs.ccsds` (codec + CRC trailer, Result-returning), new `flight.libs.commands` (typed dictionary + validator, data-not-dispatch), new enums (`LinkState`/`AckStatus`/`CommandId`/`ParamKind`), new messages (`CommandAckMsg`/`LinkStateMsg`), new config (`LinkConfig`/`CommandIngressConfig`), new `MessageType`/`FaultCode` members.

- [ ] **Step 4: Update `sim/CONTEXT.md`**: `SimStationLink` is byte-level; `build_sil_system` takes `inbound_packets: list[bytes]` + `uplink_key`; command-path SIL tests build signed packets with `build_tc_packet`; the in-process station emulator seam is the `SimStationLink` (full `packages/gse` is deferred).

- [ ] **Step 5: Write ADR 0009**

`docs/adr/0009-iss-link-transport-command-ingress.md`, Status Accepted (dated), covering: the byte-level link contract (why the link transports CCSDS bytes and iss_iface owns framing/validation -- the layered authority split); the wire format (TC = header+body+HMAC+CRC, TM = header+body+CRC); CRC-32 + HMAC-SHA256 choices; the typed-dictionary (data-not-dispatch) approach and `CommandMsg`-stays-raw decision; the always-on ACK/NACK contract; AOS/LOS gating; new fault codes as log-and-continue; HMAC key injection from the composition root; and the explicit deferral of the command router + hazardous ARM/EXECUTE to Phase 6B. Mirror the structure of ADR 0008.

- [ ] **Step 6: Add the ADR index row** to `docs/adr/README.md` (row for 0009).

- [ ] **Step 7: Gates + commit**

(Docs don't change code, but run gates to be safe -- markdown is ignored by the gates.)

```
git add packages/flight/src/flight/iss_iface/CONTEXT.md packages/flight/src/flight/hal/CONTEXT.md packages/flight/src/flight/libs/CONTEXT.md packages/sim/src/sim/CONTEXT.md docs/adr/0009-iss-link-transport-command-ingress.md docs/adr/README.md
git commit -m "docs: ADR 0009 + CONTEXT updates for ISS link transport + command ingress"
```

---

## Self-Review Checklist (run after implementing, before finishing)

1. **Spec coverage (Section 6 ingress half):** link transport (Task 6) ✓; CCSDS framing + CRC (Task 3) ✓; sequence dedup (Task 7 pipeline) ✓; HMAC auth (Task 7) ✓; typed command dictionary validation (Task 4) ✓; only validated commands become CommandMsg (Task 7) ✓; every inbound command produces ACK or NACK, always (Task 7 `_publish_ack`) ✓; AOS/LOS gating (Task 7 `pump_downlink`) ✓; link state on the bus (Task 7 `LinkStateMsg`) ✓. **Deferred (documented):** command router, ARM/EXECUTE, inhibit-at-actuation, EXIT_SAFE -> Phase 6B; storage + downlink manager -> 6C; model upload -> 6D.
2. **Per-commit green:** the Protocol change (Task 6) keeps the legacy methods so iss_iface still compiles; iss_iface migrates (Task 7); only then are the legacy methods removed (Task 8). No commit leaves the tree red.
3. **Type consistency:** `IngressOutcome` fields used identically in pipeline + app; `process_inbound` returns `(IngressOutcome, dict)` everywhere; `CommandAckMsg` field order matches Task 2; `build_apps` gains exactly one `uplink_key: bytes` param threaded by both roots; `SimStationLink` ctor `(inbound: list[bytes] | None, link_state)` matches all call sites.
4. **Invariants:** pure cores (ccsds, commands, ingress) have no bus/clock/I/O; library code returns Result/Outcome, never raises (except the test-only `build_tc_packet` helper and startup key-load); `build_apps`/scheduler stay driver-agnostic (only `main` constructs `RealStationLink`); enum values mirror names; new fault codes are log-and-continue; no peer-app cross-imports; CCSDS/commands live in libs.
5. **Placeholder scan:** no TBD/TODO left in shipped code; `_validate` is now real for the new fields. Move all inline imports in `iss_iface/app.py` to the module top before committing Task 7.
6. **Forward-compat for 6B:** `CommandSpec.hazardous` exists (unused in 6A); `CommandAckMsg` is general enough for router/target acks; the bus already carries `CommandMsg` for the future router to consume.
