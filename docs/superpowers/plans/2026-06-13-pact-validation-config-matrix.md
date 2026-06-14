# PACT Validation Configuration Matrix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring `packages/flight` validation infrastructure to a canonical configuration-matrix state: an `[environment]` config block + a single `select_drivers` factory (the only wiring change), runnable `sil` and `sil-link-real` profiles, a new `packages/gse` ground-support package (CCSDS station emulator + declarative TOML scenarios + an in-process harness backend + orchestrator/analysis), and a VCRM requirement->method->venue traceability spine with a CI check -- implemented through SIL and the `sil-link-real` x86 partial, stopping before any PIL/HIL plumbing.

**Architecture:** Validation is a configuration matrix (spec Section 9 / ADR-0010): each of {sensor, gimbal, compute, link, clock} is independently real or sim, and SIL/PIL/HIL are named corners (`profiles/*.toml`). `flight.core.select_drivers` reads `config.environment` and builds the existing driver-agnostic `Drivers` bundle; `build_apps`, the `Scheduler`, and every app are untouched. `packages/gse` (imports `flight.libs` + `sim`; never imported by them) provides the station side of `RealStationLink`'s CCSDS protocol plus a one-format / two-backend stepping seam -- only the in-process backend is implemented here. The blessed runnable partial, `sil-link-real`, drives the real CCSDS link over loopback against the GSE emulator.

**Tech Stack:** Python 3.14 (PEP 758 unparenthesized-`except` idiom), `uv` workspace, mypy `--strict`, ruff, import-linter, pytest; flight deps numpy/scipy/structlog; the gse + scenario layer uses stdlib only (`socket`, `tomllib`, `hmac`, `hashlib`, `json`).

**Inputs:** spec `docs/superpowers/specs/2026-06-09-pact-flight-final-state-design.md` Section 9 (reframed) and `docs/adr/0010-validation-configuration-matrix.md` (Accepted). Honors all CLAUDE.md invariants.

**Execution order:** Phase A -> B -> C1 -> C2 -> D, each ending on green gates. **CHECKPOINT:** stop after Phase D (SIL + `sil-link-real` green in CI, PIL/HIL profiles + procedures documented); hand back before PIL/HIL implementation (the `SocketBackend` + Jetson/bench runners).

---

## Phase A: Relocate build_tc_packet to flight.libs (Option A)

### Task A1: Move `build_tc_packet` into `flight.libs.commands.tc` and export it

**Files:**
- Create: `packages/flight/src/flight/libs/commands/tc.py`
- Modify: `packages/flight/src/flight/libs/commands/__init__.py` (lines 1-17: add `build_tc_packet` to the import block and `__all__`)

This task introduces the new canonical home for `build_tc_packet`. The body is moved **verbatim** from `flight/iss_iface/ingress/pipeline.py:67-103` (only its module-level stdlib + `flight.libs.ccsds`/`flight.libs.types` dependencies, which `flight.libs.commands` may import). It is a pure refactor; the regression guard is the existing tests `packages/flight/tests/test_iss_ingress_pipeline.py`, `packages/flight/tests/test_iss_iface_app.py`, and `packages/sim/tests/test_sil_closed_loop.py`, which must stay green after Task A2 rewires the imports. A1 in isolation only needs to (a) compile, (b) pass mypy, and (c) make the new symbol importable.

> **Python-version note (read before touching `pipeline.py`):** the repo runs Python 3.14.3 (CI is ubuntu-latest, py3.14). `pipeline.py:142` reads `except ValueError, KeyError, TypeError:` (unparenthesized exception tuple). Under **PEP 758** (new in 3.14) this is **valid** syntax — `uv run python -c "import py_compile; py_compile.compile('packages/flight/src/flight/iss_iface/ingress/pipeline.py', doraise=True)"` returns `PARSE OK`, the full non-e2e suite collects cleanly (`281 tests collected`), and the same pattern is used as-built in `flight/hal/drivers_real/station.py:67,83` (`except TimeoutError, OSError:`). **Do NOT "fix" line 142 to a parenthesized form** in this phase — it is correct, and an unrequested edit would be noise. (A 3.12-based parse would report a `SyntaxError` here; that is a false alarm against the wrong interpreter, not a real blocker.)

- [ ] **Step 1: Write the failing test**

This is a pure relocation, so the regression guard is the existing ingress/app/SIL tests (rewired in A2). To make A1 independently verifiable before A2, add one tiny import-locality test that asserts the symbol now lives at `flight.libs.commands.tc` and is re-exported from the package, and that a packet it builds round-trips through `process_inbound`.

Create `packages/flight/tests/test_commands_tc.py`:

```python
"""Locality + round-trip test for the relocated build_tc_packet helper."""

from flight.iss_iface.ingress.pipeline import process_inbound
from flight.libs.commands import build_tc_packet as build_tc_packet_pkg
from flight.libs.commands.tc import build_tc_packet
from flight.libs.types import AckStatus

_KEY = b"unit-test-key-0000000000000000000"


def test_build_tc_packet_is_exported_from_commands_package() -> None:
    """The package re-export and the submodule resolve to the same function object."""
    assert build_tc_packet_pkg is build_tc_packet


def test_build_tc_packet_roundtrips_through_process_inbound() -> None:
    """A packet built by the relocated helper is accepted by the ingress pipeline."""
    pkt = build_tc_packet("SET_THERMAL_LIMIT", {"limit_c": 70.0}, "ground", 1, _KEY, apid=1)
    outcome, _ = process_inbound(
        pkt, key=_KEY, require_auth=True, accepted_sources=("ground",), last_seq={}
    )
    assert outcome.status is AckStatus.ACCEPTED
    assert outcome.command is not None
    assert outcome.command_id == "SET_THERMAL_LIMIT"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/flight/tests/test_commands_tc.py -v`  Expected: FAIL with `ModuleNotFoundError: No module named 'flight.libs.commands.tc'` at collection (the new module does not exist yet). Note: `flight.iss_iface.ingress.pipeline` imports cleanly on Python 3.14 (line 142 is valid PEP 758), so the only collection error is the missing `tc` module — not a `SyntaxError`.

- [ ] **Step 3: Create `flight/libs/commands/tc.py` and export from the package `__init__`**

Create `packages/flight/src/flight/libs/commands/tc.py` (the `build_tc_packet` body is moved verbatim from `pipeline.py:67-103`):

```python
"""Signed CCSDS telecommand packet builder (for GSE / sim / tests; not used in flight).

Contains:
  - build_tc_packet: construct an HMAC-signed, CRC-framed TC packet from command fields.

This helper lives in flight.libs.commands so that out-of-tree command tooling (the GSE station
emulator, the SIL harness, and tests) can build authenticated telecommands while importing only
flight.libs -- never flight.iss_iface. It is the only command-side function permitted to raise
(at build/test time, on an encode failure); the runtime ingress path stays Result/Outcome-typed.

Satisfies: REQ-COMM-HIGH-003, REQ-COMM-HIGH-004.
"""

from __future__ import annotations

# stdlib
import hashlib
import hmac
import json

# internal
from flight.libs.ccsds import CcsdsHeader, encode_packet
from flight.libs.types import Err


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
        command_id: The command opcode string.
        params: The command parameters dict.
        source: The command origin identifier string.
        seq: The per-source monotonic sequence number.
        key: The shared HMAC-SHA256 secret.
        apid: The telecommand APID.

    Returns:
        The framed TC packet bytes (header + body + HMAC tag + CRC trailer).

    Notes:
        params is JSON-serialized with sorted keys so the signed bytes are deterministic.
        Raises ValueError if encode_packet rejects a field (test/build-time error only).
    """
    body = json.dumps(
        {"command_id": command_id, "params": params, "source": source, "seq": seq},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    tag = hmac.new(key, body, hashlib.sha256).digest()
    encoded = encode_packet(
        CcsdsHeader(packet_type=1, apid=apid, sequence_count=seq & 0x3FFF), body + tag
    )
    if isinstance(encoded, Err):
        raise ValueError(f"could not encode TC packet: {encoded.error}")  # test helper only
    return encoded.value
```

Replace the full contents of `packages/flight/src/flight/libs/commands/__init__.py` with (adds `build_tc_packet` to the import block and `__all__`):

```python
"""Typed command dictionary + signed-TC builder (see flight.libs.commands submodules)."""

from flight.libs.commands.dictionary import (
    COMMAND_DICTIONARY,
    CommandSpec,
    ParamSpec,
    lookup_command,
    validate_command,
)
from flight.libs.commands.tc import build_tc_packet

__all__ = [
    "COMMAND_DICTIONARY",
    "CommandSpec",
    "ParamSpec",
    "build_tc_packet",
    "lookup_command",
    "validate_command",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/flight/tests/test_commands_tc.py -v`  Expected: PASS (both tests green; the package re-export and submodule resolve to the same object and the packet round-trips). Note: at this point `pipeline.py` still defines its own `build_tc_packet`; that duplicate is removed in Task A2.

- [ ] **Step 5: Run gates**

Run: `uv run ruff check packages` -> Expected: PASS (no errors).
Run: `uv run ruff format --check packages` -> Expected: PASS (no files would be reformatted).
Run: `uv run mypy packages` -> Expected: PASS (`Success: no issues found`).
Run: `uv run lint-imports` -> Expected: PASS (all contracts kept; `flight.libs.commands.tc` imports only `flight.libs.ccsds` + `flight.libs.types`, both within `flight.libs`, so `libs-layers`/`flight-layers` hold).

- [ ] **Step 6: Commit**

```bash
git add packages/flight/src/flight/libs/commands/tc.py packages/flight/src/flight/libs/commands/__init__.py packages/flight/tests/test_commands_tc.py
git commit -m "refactor(libs): add build_tc_packet at flight.libs.commands.tc

Relocate the signed-TC builder (verbatim) into flight.libs.commands so out-of-tree
command tooling can build authenticated telecommands while importing only flight.libs.
The iss_iface pipeline copy is removed in the follow-up commit.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task A2: Delete the `pipeline.py` copy, import from libs, and rewire the three tests

**Files:**
- Modify: `packages/flight/src/flight/iss_iface/ingress/pipeline.py` (delete `def build_tc_packet` at lines 67-103; update the module docstring `Contains:` block at lines 14-17; add a re-export import + `__all__`; drop the now-unused `CcsdsHeader`/`encode_packet` imports)
- Modify: `packages/flight/src/flight/iss_iface/ingress/__init__.py` (no symbol change — it still re-exports `build_tc_packet` from `pipeline` for back-compat; verify the re-export still resolves)
- Modify: `packages/flight/tests/test_iss_ingress_pipeline.py` (line 3: switch the canonical import to `flight.libs.commands`)
- Modify: `packages/flight/tests/test_iss_iface_app.py` (line 11: switch the canonical import to `flight.libs.commands`)
- Modify: `packages/sim/tests/test_sil_closed_loop.py` (line 3: switch the canonical import to `flight.libs.commands`)

This is a **pure refactor**. The regression guard is the three existing tests named above plus `test_commands_tc.py` (from A1); all must stay green. There is no new behavioral test for A2 — the existing tests, after their imports are rewired, ARE the regression guard, plus one added assertion proving the duplicate definition is gone. After deletion, `process_inbound` no longer references `CcsdsHeader` or `encode_packet` (both were used **only** by the deleted `build_tc_packet`); `decode_packet`, `hashlib`, `hmac`, and `json` remain in use. `pipeline.py` then re-exports `build_tc_packet` from `flight.libs.commands` so `flight.iss_iface.ingress` keeps the back-compat name.

> **Do NOT touch `pipeline.py:142`.** `except ValueError, KeyError, TypeError:` is valid Python 3.14 (PEP 758) and the file parses and imports cleanly (see the version note in A1). The only edits in this task are the docstring block, the import block, and the `build_tc_packet` deletion.

- [ ] **Step 1: Write the failing test (rewire the existing tests to the canonical path)**

The three existing tests are the guard. Rewire their `build_tc_packet` import from `flight.iss_iface.ingress` to the new canonical `flight.libs.commands`, and ALSO add an assertion to `test_iss_ingress_pipeline.py` proving `pipeline.py` no longer defines its own copy (it must resolve to the libs object). Apply these exact edits.

In `packages/flight/tests/test_iss_ingress_pipeline.py`, replace line 3:

```python
from flight.iss_iface.ingress import build_tc_packet, process_inbound
```
with:
```python
from flight.iss_iface.ingress import process_inbound
from flight.libs.commands import build_tc_packet
```

In `packages/flight/tests/test_iss_iface_app.py`, replace line 11:

```python
from flight.iss_iface.ingress import build_tc_packet
```
with:
```python
from flight.libs.commands import build_tc_packet
```

In `packages/sim/tests/test_sil_closed_loop.py`, replace line 3:

```python
from flight.iss_iface.ingress import build_tc_packet
```
with:
```python
from flight.libs.commands import build_tc_packet
```

Append this regression test to the end of `packages/flight/tests/test_iss_ingress_pipeline.py` (proves the duplicate definition is gone and the back-compat re-export still works):

```python
def test_pipeline_no_longer_defines_build_tc_packet() -> None:
    """The pipeline module re-exports the libs builder rather than defining its own copy."""
    import flight.iss_iface.ingress as ingress
    import flight.iss_iface.ingress.pipeline as pipeline
    from flight.libs.commands import build_tc_packet as libs_build_tc_packet

    assert pipeline.build_tc_packet is libs_build_tc_packet
    assert ingress.build_tc_packet is libs_build_tc_packet
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/flight/tests/test_iss_ingress_pipeline.py::test_pipeline_no_longer_defines_build_tc_packet -v`  Expected: FAIL — `pipeline.build_tc_packet` is still the locally-defined function (lines 67-103), so `pipeline.build_tc_packet is libs_build_tc_packet` is `False` (`assert` raises `AssertionError`). The module imports fine (no `SyntaxError`); the failure is the intended `AssertionError`.

- [ ] **Step 3: Delete the duplicate in `pipeline.py`, re-export from libs, and fix the docstring/imports**

In `packages/flight/src/flight/iss_iface/ingress/pipeline.py`, update the module docstring `Contains:` block (lines 14-17) from:

```python
Contains:
  - IngressOutcome: the per-packet result (command-or-None + ack status + reason + echo).
  - build_tc_packet: construct a signed TC packet (used by GSE/sim/tests, not flight).
  - process_inbound: run the full pipeline for one raw packet (Result-free; outcome-typed).
```
to:
```python
Contains:
  - IngressOutcome: the per-packet result (command-or-None + ack status + reason + echo).
  - process_inbound: run the full pipeline for one raw packet (Result-free; outcome-typed).

build_tc_packet (the signed-TC builder for GSE/sim/tests) now lives in flight.libs.commands.tc
and is re-exported here for back-compat.
```

Replace the import block (lines 22-34). The current block is:

```python
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
```

Replace it with (drop `CcsdsHeader` + `encode_packet` from the ccsds import — `process_inbound` uses only `decode_packet`; add the `build_tc_packet` re-export to the commands import; add an explicit `__all__`):

```python
from __future__ import annotations

# stdlib
import hashlib
import hmac
import json
from dataclasses import dataclass

# internal
from flight.libs.ccsds import decode_packet
from flight.libs.commands import build_tc_packet, lookup_command, validate_command
from flight.libs.messages import CommandMsg
from flight.libs.types import AckStatus, Err, FaultCode, MessageType

__all__ = ["IngressOutcome", "build_tc_packet", "process_inbound"]
```

Rationale for the dropped imports (verified against the file): after `build_tc_packet` is deleted, the `process_inbound` body (lines 106-186) references `decode_packet` (1x), `hmac.new`/`hmac.compare_digest` (4x), `hashlib.sha256` (1x), and `json.loads` (1x) — but references `CcsdsHeader` **0x** and `encode_packet` **0x** (both were used only by the deleted builder). Keeping either would trip Ruff `F401` (unused import). All of `hashlib`, `hmac`, `json`, `Err`, `AckStatus`, `FaultCode`, `MessageType`, `CommandMsg`, `lookup_command`, and `validate_command` remain used by `process_inbound`/`_reject`/`IngressOutcome`, so they stay. The `__all__` makes the `build_tc_packet` re-export explicit so Ruff's unused-import lint (F401) does not flag it; `pipeline.py` defines only the public names `IngressOutcome`, `build_tc_packet`, and `process_inbound` (plus the private `_reject`/`_HMAC_TAG_SIZE`), so listing those three is the complete and safe public surface — matching the `__all__` style already used in `flight/iss_iface/ingress/__init__.py` and `flight/libs/commands/__init__.py`.

Delete the entire `build_tc_packet` definition (current lines 67-103, including the two blank lines 65-66 that precede it after `_reject`). Concretely, remove this block:

```python
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
        command_id: The command opcode string.
        params: The command parameters dict.
        source: The command origin identifier string.
        seq: The per-source monotonic sequence number.
        key: The shared HMAC-SHA256 secret.
        apid: The telecommand APID.

    Returns:
        The framed TC packet bytes (header + body + HMAC tag + CRC trailer).

    Notes:
        params is JSON-serialized with sorted keys so the signed bytes are deterministic.
        Raises ValueError if encode_packet rejects a field (test/build-time error only).
    """
    body = json.dumps(
        {"command_id": command_id, "params": params, "source": source, "seq": seq},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    tag = hmac.new(key, body, hashlib.sha256).digest()
    encoded = encode_packet(
        CcsdsHeader(packet_type=1, apid=apid, sequence_count=seq & 0x3FFF), body + tag
    )
    if isinstance(encoded, Err):
        raise ValueError(f"could not encode TC packet: {encoded.error}")  # test helper only
    return encoded.value
```

so that the `_reject` helper (ending at line 64) is followed by exactly two blank lines and then `def process_inbound(` (no remaining `build_tc_packet` body). The `flight/iss_iface/ingress/__init__.py` re-export (`from flight.iss_iface.ingress.pipeline import IngressOutcome, build_tc_packet, process_inbound`) is unchanged and continues to work because `pipeline` now re-exports the name from libs.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/flight/tests/test_iss_ingress_pipeline.py::test_pipeline_no_longer_defines_build_tc_packet -v`  Expected: PASS (`pipeline.build_tc_packet` and `ingress.build_tc_packet` are now the libs object).

Run the full regression set (the pure-refactor guards):
Run: `uv run pytest packages/flight/tests/test_iss_ingress_pipeline.py packages/flight/tests/test_iss_iface_app.py packages/flight/tests/test_commands_tc.py packages/sim/tests/test_sil_closed_loop.py -v`  Expected: PASS (all tests green; behavior is unchanged, only the import path moved).

- [ ] **Step 5: Run gates**

Run: `uv run ruff check packages` -> Expected: PASS. No `F401` on `build_tc_packet` (the `__all__` re-exports it); and no `F401` on `CcsdsHeader`/`encode_packet` because both were dropped from the import block (they were used only by the deleted builder). `decode_packet`, `hashlib`, `hmac`, and `json` remain referenced by `process_inbound`.
Run: `uv run ruff format --check packages` -> Expected: PASS (no files would be reformatted).
Run: `uv run mypy packages` -> Expected: PASS (`Success: no issues found`).
Run: `uv run lint-imports` -> Expected: PASS (`flight.iss_iface.ingress.pipeline` now imports `build_tc_packet` from `flight.libs.commands`, which is in a lower layer than the apps — `flight-layers` still holds; no app cross-import introduced).
Run: `uv run pytest packages -m "not e2e"` -> Expected: PASS (full non-e2e suite green; confirms no other module relied on `build_tc_packet` living in `pipeline.py`).

- [ ] **Step 6: Commit**

```bash
git add packages/flight/src/flight/iss_iface/ingress/pipeline.py packages/flight/tests/test_iss_ingress_pipeline.py packages/flight/tests/test_iss_iface_app.py packages/sim/tests/test_sil_closed_loop.py
git commit -m "refactor(iss_iface): import build_tc_packet from flight.libs.commands

Remove the duplicate build_tc_packet definition from the ingress pipeline and re-export
the canonical flight.libs.commands.tc copy for back-compat. Drop the now-unused CcsdsHeader
and encode_packet imports (process_inbound uses only decode_packet). Rewire the three
command-path tests to the canonical import. Pure refactor: ingress/app/SIL tests stay green.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

## Phase B: [environment] config + select_drivers + profiles

### Task B1: Add `[environment]` config (AxisMode + EnvironmentConfig) to config.py, default.toml, exports, and config_loader mapping

**Files:**
- Modify: `packages/flight/src/flight/libs/config/config.py` (add `AxisMode`, `EnvironmentConfig`; add `environment` field to `PactConfig`, lines 194-211)
- Modify: `config/default.toml` (append `[environment]` block after line 107)
- Modify: `packages/flight/src/flight/libs/config/__init__.py` (re-export `AxisMode`, `EnvironmentConfig`)
- Modify: `packages/flight/src/flight/core/config_loader.py` (import `AxisMode` + `EnvironmentConfig`; add `_axis_mode` helper + `env` mapping; add `environment=` to returned `PactConfig`)
- Modify/Test: `packages/flight/tests/test_config_defaults.py` (add `environment` to `_SECTION_TO_DATACLASS`)
- Create/Test: `packages/flight/tests/test_config_environment.py`

- [ ] **Step 1: Write the failing test**

Create `packages/flight/tests/test_config_environment.py`:

```python
"""Verifies the [environment] config axis defaults and loader mapping."""

import dataclasses
import tomllib
from pathlib import Path

from flight.core.config_loader import load_config
from flight.libs.config import EnvironmentConfig, PactConfig
from flight.libs.types import Ok

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_TOML = _REPO_ROOT / "config" / "default.toml"


def test_environment_defaults_all_real() -> None:
    """A bare EnvironmentConfig has every axis 'real' and the jetson host."""
    env = EnvironmentConfig()
    assert (env.sensor, env.gimbal, env.compute, env.link, env.clock) == (
        "real",
        "real",
        "real",
        "real",
        "real",
    )
    assert env.host == "jetson_aarch64"


def test_pactconfig_has_environment_field_last() -> None:
    """PactConfig exposes an environment field, declared last."""
    field_names = [f.name for f in dataclasses.fields(PactConfig)]
    assert field_names[-1] == "environment"
    assert PactConfig().environment == EnvironmentConfig()


def test_environment_defaults_match_default_toml() -> None:
    """The [environment] section of default.toml equals the dataclass defaults."""
    with _DEFAULT_TOML.open("rb") as fh:
        toml_data = tomllib.load(fh)
    section = toml_data["environment"]
    defaults = EnvironmentConfig()
    for field in dataclasses.fields(EnvironmentConfig):
        assert section[field.name] == getattr(defaults, field.name), field.name


def test_loader_maps_environment_section() -> None:
    """load_config populates environment from default.toml (all axes real)."""
    result = load_config(str(_DEFAULT_TOML))
    assert isinstance(result, Ok)
    assert result.value.environment == EnvironmentConfig()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/flight/tests/test_config_environment.py -v`
Expected: FAIL with `ImportError: cannot import name 'EnvironmentConfig' from 'flight.libs.config'`.

- [ ] **Step 3: Implement the config dataclass, TOML block, exports, and loader mapping**

In `packages/flight/src/flight/libs/config/config.py`, change the imports (lines 14-17) to add `Literal`:

```python
from __future__ import annotations

# stdlib
from dataclasses import dataclass, field
from typing import Literal
```

Add, immediately above the `PactConfig` definition (i.e. before line 194 `@dataclass(frozen=True)\nclass PactConfig`):

```python
# A deployment axis is wired to either a sim stand-in or the real device/driver.
AxisMode = Literal["sim", "real"]


@dataclass(frozen=True)
class EnvironmentConfig:
    """Per-axis sim/real wiring selector for the composition root.

    Each field names a deployment axis the composition root must resolve to a
    concrete driver: 'sim' selects an in-process stand-in, 'real' selects the
    flight driver/device. host is a free-form label for the target machine
    (provenance only; not acted on). The 'lock' (LaunchLock) axis is intentionally
    absent: there is no LaunchLock device, so it is a permanent VCRM gap, not a
    config field. The clock axis is informational here -- the composition root
    chooses RealClock vs ManualClock from it BEFORE building drivers.

    Satisfies: REQ-OPER-HIGH-002 (validated startup config selects the deployment axes).
    """

    sensor: AxisMode = "real"  # imaging sensor: SimSensor vs RealSensor
    gimbal: AxisMode = "real"  # gimbal actuator: SimGimbal vs RealGimbal
    compute: AxisMode = "real"  # detector backend: ScriptedDetector vs OnnxDetector
    link: AxisMode = "real"  # station link: SimStationLink vs RealStationLink
    clock: AxisMode = "real"  # ManualClock (sim) vs RealClock (real); read by the root
    host: str = "jetson_aarch64"  # target-machine label (provenance only)
```

Add the `environment` field as the LAST field of `PactConfig` (after line 211 `command_ingress=...`):

```python
    command_ingress: CommandIngressConfig = field(default_factory=CommandIngressConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
```

In `config/default.toml`, append after line 107 (the `accepted_sources` line of `[command_ingress]`):

```toml

[environment]
sensor = "real"
gimbal = "real"
compute = "real"
link = "real"
clock = "real"
host = "jetson_aarch64"
```

In `packages/flight/src/flight/libs/config/__init__.py`, add `AxisMode` and `EnvironmentConfig` to the import and `__all__`:

```python
"""Typed, frozen flight configuration dataclasses.

Each subsystem receives its own sub-config. Defaults here MUST match
config/default.toml (enforced by tests/test_config_defaults.py).
"""

from flight.libs.config.config import (
    AxisMode,
    CommandIngressConfig,
    CommsConfig,
    ControllerConfig,
    EnvironmentConfig,
    FaultConfig,
    GimbalConfig,
    InferenceConfig,
    LinkConfig,
    PactConfig,
    PreprocessingConfig,
    SensorConfig,
    StorageConfig,
)

__all__ = [
    "AxisMode",
    "CommsConfig",
    "CommandIngressConfig",
    "ControllerConfig",
    "EnvironmentConfig",
    "FaultConfig",
    "GimbalConfig",
    "InferenceConfig",
    "LinkConfig",
    "PactConfig",
    "PreprocessingConfig",
    "SensorConfig",
    "StorageConfig",
]
```

In `packages/flight/src/flight/core/config_loader.py`, extend the config import block (lines 19-31) to add BOTH `AxisMode` (used by the `_axis_mode` helper's return annotation) and `EnvironmentConfig` (the helper feeds it). Add both in this single edit -- do NOT split this into two separate edits, or one will be a redundant no-op:

```python
from flight.libs.config import (
    AxisMode,
    CommandIngressConfig,
    CommsConfig,
    ControllerConfig,
    EnvironmentConfig,
    FaultConfig,
    GimbalConfig,
    InferenceConfig,
    LinkConfig,
    PactConfig,
    PreprocessingConfig,
    SensorConfig,
    StorageConfig,
)
```

(`Any` is already imported at line 16, so the helper's `section: dict[str, Any]` parameter needs no new typing import.)

Add the `_axis_mode` helper as a MODULE-LEVEL function (not nested) immediately above `_build_pact_config` (before line 119 `def _build_pact_config`); the explicit `if`/`if`/`raise` branches -- not a `cast` -- are what keep the return statically narrowed to `AxisMode` under mypy --strict:

```python
def _axis_mode(section: dict[str, Any], key: str, default: str) -> AxisMode:
    """Resolve and validate one environment axis to the 'sim'/'real' literal.

    Args:
        section: The parsed [environment] TOML dict (or {} when absent).
        key: The axis field name (e.g. "sensor").
        default: The dataclass default for the axis ("sim" or "real").

    Returns:
        The validated AxisMode literal ("sim" or "real").

    Raises:
        ValueError: If the configured value is neither "sim" nor "real". Explicit
            branches (no cast) keep the return statically typed as AxisMode under
            mypy --strict.
    """
    raw = str(section.get(key, default))
    if raw == "sim":
        return "sim"
    if raw == "real":
        return "real"
    raise ValueError(f"environment.{key} must be 'sim' or 'real', got {raw!r}")
```

Inside `_build_pact_config`, add the `env` mapping immediately before the final `return PactConfig(...)` (the live file's `return PactConfig(` is at line 319):

```python
    env = data.get("environment", {})
    environment_config = EnvironmentConfig(
        sensor=_axis_mode(env, "sensor", EnvironmentConfig.sensor),
        gimbal=_axis_mode(env, "gimbal", EnvironmentConfig.gimbal),
        compute=_axis_mode(env, "compute", EnvironmentConfig.compute),
        link=_axis_mode(env, "link", EnvironmentConfig.link),
        clock=_axis_mode(env, "clock", EnvironmentConfig.clock),
        host=str(env.get("host", EnvironmentConfig.host)),
    )
```

Add `environment=environment_config` as the last argument of the returned `PactConfig` (after `command_ingress=command_ingress_config,`):

```python
    return PactConfig(
        controller=controller_config,
        inference=inference_config,
        comms=comms_config,
        storage=storage_config,
        fault=fault_config,
        preprocessing=preprocessing_config,
        sensor=sensor_config,
        gimbal=gimbal_config,
        link=link_config,
        command_ingress=command_ingress_config,
        environment=environment_config,
    )
```

Finally, extend the defaults test so the new section is guarded (this test plus `test_loader_maps_environment_section` jointly guard the config.py-defaults == default.toml invariant for the new section). In `packages/flight/tests/test_config_defaults.py`, add the import and the section-map entry:

```python
from flight.libs.config import (
    CommandIngressConfig,
    CommsConfig,
    ControllerConfig,
    EnvironmentConfig,
    FaultConfig,
    GimbalConfig,
    InferenceConfig,
    LinkConfig,
    PreprocessingConfig,
    SensorConfig,
    StorageConfig,
)
```

```python
_SECTION_TO_DATACLASS = {
    "controller": ControllerConfig,
    "inference": InferenceConfig,
    "comms": CommsConfig,
    "storage": StorageConfig,
    "preprocessing": PreprocessingConfig,
    "fault": FaultConfig,
    "sensor": SensorConfig,
    "gimbal": GimbalConfig,
    "link": LinkConfig,
    "command_ingress": CommandIngressConfig,
    "environment": EnvironmentConfig,
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/flight/tests/test_config_environment.py packages/flight/tests/test_config_defaults.py packages/flight/tests/test_config_loader.py -v`
Expected: PASS (all environment tests green; `test_config_defaults_match_default_toml` still green with the new `environment` section; the loader-mapping test confirms all five axes + host round-trip exactly, so a forgotten axis in `_build_pact_config` would fail here).

- [ ] **Step 5: Run gates**

Run each and expect clean:
- `uv run ruff check packages`
- `uv run ruff format --check packages`
- `uv run mypy packages`  (the explicit `_axis_mode` branches keep the `AxisMode` literal narrow under --strict)
- `uv run lint-imports`
- `uv run pytest packages -m "not e2e"`

- [ ] **Step 6: Commit**

```bash
git add packages/flight/src/flight/libs/config/config.py config/default.toml packages/flight/src/flight/libs/config/__init__.py packages/flight/src/flight/core/config_loader.py packages/flight/tests/test_config_defaults.py packages/flight/tests/test_config_environment.py && git commit -m "feat(config): add [environment] axis selector (sim/real per axis)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task B2: Create `flight/core/select_drivers.py` (SimDriverInputs + env-driven select_drivers)

**Files:**
- Create: `packages/flight/src/flight/core/select_drivers.py`
- Create/Test: `packages/flight/tests/test_select_drivers.py`

- [ ] **Step 1: Write the failing test**

Create `packages/flight/tests/test_select_drivers.py`:

```python
"""Verifies env-driven driver selection: all-sim, missing sim_inputs, link=real."""

import dataclasses
import socket

import pytest

from flight.core.select_drivers import SimDriverInputs, select_drivers
from flight.hal.drivers_real import RealStationLink
from flight.hal.drivers_sim import SimGimbal, SimScalarSensor, SimSensor, SimStationLink
from flight.libs.config import PactConfig
from flight.libs.time import ManualClock
from sim.scene import build_frames, plume_detector


def _all_sim_config() -> PactConfig:
    """A PactConfig with every environment axis forced to 'sim'."""
    base = PactConfig()
    env = dataclasses.replace(
        base.environment,
        sensor="sim",
        gimbal="sim",
        compute="sim",
        link="sim",
        clock="sim",
    )
    return dataclasses.replace(base, environment=env)


def _sim_inputs() -> SimDriverInputs:
    """A populated SimDriverInputs for the all-sim path."""
    return SimDriverInputs(
        frames=build_frames(2),
        detector=plume_detector(),
        inbound_packets=[],
        thermal_readings=[25.0],
        power_readings=[30.0],
    )


def test_all_sim_returns_sim_drivers_and_passed_detector() -> None:
    """All-sim selection wires every sim driver and reuses the passed detector."""
    inputs = _sim_inputs()
    drivers = select_drivers(_all_sim_config(), ManualClock(), inputs)
    assert isinstance(drivers.sensor, SimSensor)
    assert isinstance(drivers.gimbal, SimGimbal)
    assert isinstance(drivers.station, SimStationLink)
    assert isinstance(drivers.thermal_sensor, SimScalarSensor)
    assert isinstance(drivers.power_sensor, SimScalarSensor)
    assert drivers.detector is inputs.detector


def test_sim_axis_without_inputs_raises() -> None:
    """A sim axis with sim_inputs=None is a programming error -> ValueError."""
    with pytest.raises(ValueError, match="sim_inputs"):
        select_drivers(_all_sim_config(), ManualClock(), None)


def test_link_real_builds_realstationlink() -> None:
    """link='real' (others sim) builds a RealStationLink bound to a free port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        free_port = probe.getsockname()[1]

    base = _all_sim_config()
    env = dataclasses.replace(base.environment, link="real")
    link_cfg = dataclasses.replace(base.link, command_tcp_port=free_port)
    config = dataclasses.replace(base, environment=env, link=link_cfg)

    drivers = select_drivers(config, ManualClock(), _sim_inputs())
    try:
        assert isinstance(drivers.station, RealStationLink)
        assert isinstance(drivers.sensor, SimSensor)
    finally:
        drivers.station.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/flight/tests/test_select_drivers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'flight.core.select_drivers'`.

- [ ] **Step 3: Implement `select_drivers.py`**

Create `packages/flight/src/flight/core/select_drivers.py`. Each branch local is annotated with its HAL Protocol (`ImagingSensor`, `GimbalActuator`, `DetectorBackend`, `StationLink`, `ScalarSensor`) imported at module top from `flight.hal.interfaces` (+ `DetectorBackend` from `flight.payload.model`). These interface modules are pure-Protocol and SDK-free, so importing them at module top does NOT break the lazy-SDK invariant; only the concrete `drivers_real.*` classes stay lazily imported inside their `real` branches. Both sim and real concrete classes implement these Protocols structurally, so the assignments and the final `Drivers(...)` construction type-check under mypy --strict with NO `# type: ignore`:

```python
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
from flight.hal.drivers_sim import SimGimbal, SimScalarSensor, SimSensor, SimStationLink
from flight.hal.interfaces import (
    GimbalActuator,
    ImagingSensor,
    ScalarSensor,
    StationLink,
)
from flight.libs.config import PactConfig
from flight.libs.time import Clock
from flight.libs.types import MosaicFrame, Ok
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

        detector = OnnxDetector(config.inference.model_path)

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
        detector=detector,
        station=station,
        thermal_sensor=thermal_sensor,
        power_sensor=power_sensor,
    )
```

Note on typing: `Drivers` fields are Protocol-typed (`ImagingSensor`, `GimbalActuator`, `DetectorBackend`, `StationLink`, `ScalarSensor`). Because those Protocols are imported at module top from `flight.hal.interfaces`/`flight.payload.model` (both SDK-free pure-Protocol modules) and each branch local is annotated with its Protocol, mypy --strict checks every assignment AND the `Drivers(...)` call directly -- no `# type: ignore` and no `cast`. The lazily-imported concrete real classes (`RealSensor`, `RealGimbal`, `RealStationLink`, `RealScalarSensor`, `OnnxDetector`) are assigned into the already-Protocol-typed locals, so their structural conformance is verified at the assignment site without naming them at module top, preserving the lazy-SDK invariant.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/flight/tests/test_select_drivers.py -v`
Expected: PASS (3 tests; the link=real test binds a free port, asserts `RealStationLink`, then closes it).

- [ ] **Step 5: Run gates**

Run each and expect clean:
- `uv run ruff check packages`
- `uv run ruff format --check packages`
- `uv run mypy packages`  (every branch local is Protocol-typed; the assignments and the `Drivers(...)` call are checked with no suppression)
- `uv run lint-imports`  (CRITICAL: `select_drivers` imports `drivers_real`/`drivers_sim` but lives in `flight.core`, which is NOT a source module of `drivers-from-composition-roots-only`, so the contract still passes; importing the pure-Protocol `flight.hal.interfaces` is allowed under `flight-layers`)
- `uv run pytest packages -m "not e2e"`

- [ ] **Step 6: Commit**

```bash
git add packages/flight/src/flight/core/select_drivers.py packages/flight/tests/test_select_drivers.py && git commit -m "feat(core): add env-driven select_drivers (sim/real per axis)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task B3: Refactor `sim/sil/runner.py` build_sil_system to delegate to select_drivers

**Files:**
- Modify: `packages/sim/src/sim/sil/runner.py` (lines 18-101: imports + `build_sil_system` body)
- Regression guard: `packages/sim/tests/test_sil_closed_loop.py` (all ~9 existing tests; unchanged)

- [ ] **Step 1: Confirm the regression guard (existing tests are the spec)**

This is a pure refactor: `build_sil_system` must keep returning a `SilSystem` whose `sensor`/`gimbal`/`station`/`thermal_sensor`/`power_sensor` are the concrete sim types and whose apps behave identically. The guard is the existing `packages/sim/tests/test_sil_closed_loop.py` (e.g. `test_sil_nominal_closed_loop_tracks_plume`, the SAFE-on-thermal, command-ingress, AOS/LOS tests). No new test is added; these must stay green unchanged.

- [ ] **Step 2: Run the guard to confirm it currently passes (pre-refactor baseline)**

Run: `uv run pytest packages/sim/tests/test_sil_closed_loop.py -v`
Expected: PASS (baseline; ~9 tests green before the change).

- [ ] **Step 3: Refactor `build_sil_system` to delegate to `select_drivers`**

Replace the imports block and the `build_sil_system` body in `packages/sim/src/sim/sil/runner.py`. Change the import region (lines 16-32) to:

```python
from __future__ import annotations

# stdlib
import dataclasses
from dataclasses import dataclass
from typing import cast

# internal
from flight.core.composition import MONITORED_SUBSYSTEMS, SystemApps, build_apps
from flight.core.select_drivers import SimDriverInputs, select_drivers
from flight.fault.watchdog import WatchdogEntry
from flight.hal.drivers_sim import SimGimbal, SimScalarSensor, SimSensor, SimStationLink
from flight.libs.bus import MessageBus
from flight.libs.config import EnvironmentConfig, PactConfig
from flight.libs.messages import HeartbeatMsg
from flight.libs.time import ManualClock
from flight.libs.types import GimbalState, MessageType, MosaicFrame, Ok
from flight.payload.calibration_io import build_identity_calibration
from flight.payload.control import ControlState
from flight.payload.model import ScriptedDetector
```

(`Drivers` is no longer constructed directly, so it is dropped from the composition import; `dataclasses` and `cast` are added.)

Replace the body of `build_sil_system` (lines 76-101, from `bus = MessageBus()` through the closing `return SilSystem(...)`) with:

```python
    bus = MessageBus()
    sim_inputs = SimDriverInputs(
        frames=frames,
        detector=detector,
        inbound_packets=inbound_packets or [],
        thermal_readings=thermal_readings or [],
        power_readings=power_readings or [],
    )
    sil_env = EnvironmentConfig(
        sensor="sim",
        gimbal="sim",
        compute="sim",
        link="sim",
        clock="sim",
        host="x86_64",
    )
    sil_config = dataclasses.replace(config, environment=sil_env)
    drivers = select_drivers(sil_config, clock, sim_inputs)
    calib = build_identity_calibration(config.sensor.height_px, config.sensor.width_px)
    apps = build_apps(sil_config, bus, clock, drivers, MONITORED_SUBSYSTEMS, calib, uplink_key)
    return SilSystem(
        apps=apps,
        bus=bus,
        clock=clock,
        sensor=cast(SimSensor, drivers.sensor),
        gimbal=cast(SimGimbal, drivers.gimbal),
        station=cast(SimStationLink, drivers.station),
        thermal_sensor=cast(SimScalarSensor, drivers.thermal_sensor),
        power_sensor=cast(SimScalarSensor, drivers.power_sensor),
    )
```

Update the `build_sil_system` docstring to record the delegation (insert into the existing docstring, after the `Returns:` block):

```python
    Notes:
        Delegates concrete-driver construction to flight.core.select_drivers with an
        all-"sim" EnvironmentConfig (host "x86_64"), so the SIL exercises the exact
        same selection path the flight entry uses. The returned SilSystem casts the
        Protocol-typed Drivers fields back to their concrete sim types for inspection.
    """
```

The direct `SimGimbal(clock=...)`/`SimSensor(...)`/`SimStationLink(...)`/`SimScalarSensor(...)` constructions are removed (now handled by `select_drivers`); confirm no other reference to `Drivers` remains in the file after dropping it from the import.

- [ ] **Step 4: Run the guard to verify it still passes**

Run: `uv run pytest packages/sim/tests/test_sil_closed_loop.py -v`
Expected: PASS (all ~9 tests green, unchanged behavior).

- [ ] **Step 5: Run gates**

Run each and expect clean:
- `uv run ruff check packages`
- `uv run ruff format --check packages`
- `uv run mypy packages`  (the `cast(...)` calls narrow the Protocol-typed `Drivers` fields back to the concrete sim types `SilSystem` declares)
- `uv run lint-imports`  (sim still only reaches flight via `flight.core.*` + Protocols; the `drivers-from-composition-roots-only` contract is unaffected since `sim` is not one of its source modules)
- `uv run pytest packages -m "not e2e"`

- [ ] **Step 6: Commit**

```bash
git add packages/sim/src/sim/sil/runner.py && git commit -m "refactor(sil): build_sil_system delegates driver construction to select_drivers

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task B4: Wire `flight/core/main.py` build_flight_system to obtain Drivers from select_drivers

**Files:**
- Modify: `packages/flight/src/flight/core/main.py` (lines 20-101 imports + `build_flight_system`; lines 129-131 clock selection in `main`)

- [ ] **Step 1: No unit test (real drivers absent in CI) -- state the verification strategy**

`build_flight_system` constructs real drivers (PySpin/pyserial/onnxruntime/socket) and cannot run in CI. The verification for this task is `uv run mypy packages` plus `uv run lint-imports` (proving the rewired imports and the `select_drivers` call type-check and respect the driver-from-composition-root contract), and the existing SIL/loader suites staying green. No new pytest test is added; this is recorded as a deliberate gap (real-hardware build is HIL-only).

- [ ] **Step 2: Confirm baseline mypy + import gates are clean before the change**

Run: `uv run mypy packages` and `uv run lint-imports`
Expected: PASS (clean baseline).

- [ ] **Step 3: Rewire `build_flight_system` to use `select_drivers`; pick the clock from the env**

In `packages/flight/src/flight/core/main.py`, replace the import block (lines 20-31). The concrete real-driver classes (`RealGimbal`, `RealScalarSensor`, `RealSensor`, `RealStationLink`) and `OnnxDetector` are no longer constructed here -- `select_drivers` owns that, lazily -- so drop those imports entirely. `Drivers` is ALSO dropped (it is no longer constructed in this module; the rewritten `build_flight_system` calls `select_drivers(...)` and passes the returned bundle straight into `build_apps`, never naming the `Drivers` type). Likewise `SimDriverInputs` is NOT imported here: the default flight env has no 'sim' axis, so `select_drivers` is called with `sim_inputs=None`. Add only `select_drivers`, and broaden the time import to `Clock, ManualClock, RealClock`:

```python
# internal
from flight.core.composition import MONITORED_SUBSYSTEMS, SystemApps, build_apps
from flight.core.config_loader import load_config
from flight.core.scheduler import Scheduler
from flight.core.select_drivers import select_drivers
from flight.libs.bus import MessageBus
from flight.libs.config import PactConfig
from flight.libs.time import Clock, ManualClock, RealClock
from flight.libs.types import Ok
from flight.payload.calibration_io import build_identity_calibration, load_calibration
from flight.payload.preprocess import MosaicCalibration
```

(Importing `Drivers` or `SimDriverInputs` here would be an unused import and fail the `ruff check` F401 gate -- the rewritten body below references neither type.)

Replace the entire `build_flight_system` function (lines 54-101) with:

```python
def build_flight_system(
    config: PactConfig, bus: MessageBus, clock: Clock, calib: MosaicCalibration
) -> SystemApps:
    """Resolve the env-selected Drivers bundle and wire the SystemApps.

    Args:
        config: The validated PactConfig (its environment axes select each driver).
        bus: The shared MessageBus.
        clock: The injected Clock (chosen in main from config.environment.clock).
        calib: The MosaicCalibration to inject into the payload app (loaded from
            checksummed artifacts, or identity when no calibration_dir is configured).

    Returns:
        The wired SystemApps.

    Raises:
        SystemExit: If the uplink key file is missing/unreadable, or if the
            real-sensor startup exposure/gain tuning fails (both unrecoverable at
            startup; the latter now lives inside select_drivers).
        ValueError: If a 'real' gimbal is selected with an empty config.gimbal.serial_port
            (RealGimbal cannot open its link -- an unrecoverable startup misconfig).

    Notes:
        Driver construction is delegated to flight.core.select_drivers, which lazily
        imports PySpin/pyserial/onnxruntime only inside the 'real' branches it backs.
        With the default all-"real" environment this builds the full hardware stack, so
        this function runs only on flight hardware. sim_inputs is None: the default flight
        env has no 'sim' axis, so no sim construction inputs are needed (select_drivers
        raises ValueError if that assumption is ever violated by a misconfigured env).
    """
    uplink_key = _load_uplink_key(config.command_ingress.hmac_key_path)
    drivers = select_drivers(config, clock, sim_inputs=None)
    return build_apps(config, bus, clock, drivers, MONITORED_SUBSYSTEMS, calib, uplink_key)
```

(The duplicate exposure/gain block that was in `build_flight_system` is removed -- it now lives once in `select_drivers`.)

In `main`, replace the fixed `clock: Clock = RealClock()` (line 130) with an env-driven choice:

```python
    bus = MessageBus()
    clock: Clock = RealClock() if config.environment.clock == "real" else ManualClock()
    apps = build_flight_system(config, bus, clock, calib)
```

- [ ] **Step 4: Verify (no runtime test -- mypy + imports stand in)**

Run: `uv run mypy packages`
Expected: PASS (the rewired `build_flight_system` and clock selection type-check; `ManualClock`/`RealClock` both satisfy the `Clock` Protocol).

- [ ] **Step 5: Run gates**

Run each and expect clean:
- `uv run ruff check packages`  (no unused imports: `Drivers`/`SimDriverInputs`/the concrete real drivers/`OnnxDetector` are all gone from this module)
- `uv run ruff format --check packages`
- `uv run mypy packages`
- `uv run lint-imports`  (`main` no longer imports the concrete real drivers directly -- they are reached only transitively through `select_drivers`, which is allowed for `flight.core`)
- `uv run pytest packages -m "not e2e"`  (existing loader/SIL suites still green; no new test)

- [ ] **Step 6: Commit**

```bash
git add packages/flight/src/flight/core/main.py && git commit -m "refactor(core): build_flight_system uses select_drivers; main picks clock from env

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task B5: Create the deployment profiles (sil, sil-link-real, pil, hil)

**Files:**
- Create: `profiles/sil.toml`
- Create: `profiles/sil-link-real.toml`
- Create: `profiles/pil.toml`
- Create: `profiles/hil.toml`
- Create/Test: `packages/flight/tests/test_profiles.py`

- [ ] **Step 1: Write the failing test**

Create `packages/flight/tests/test_profiles.py`:

```python
"""Verifies the deployment profiles override the [environment] axes correctly."""

from pathlib import Path

import pytest

from flight.core.config_loader import load_config
from flight.libs.types import Ok

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT = str(_REPO_ROOT / "config" / "default.toml")


def _profile(name: str) -> str:
    """Absolute path to a repo-root profile override."""
    return str(_REPO_ROOT / "profiles" / name)


def test_sil_profile_all_axes_sim() -> None:
    """profiles/sil.toml forces every environment axis to 'sim'."""
    result = load_config(_DEFAULT, _profile("sil.toml"))
    assert isinstance(result, Ok)
    env = result.value.environment
    assert (env.sensor, env.gimbal, env.compute, env.link, env.clock) == (
        "sim",
        "sim",
        "sim",
        "sim",
        "sim",
    )
    assert env.host == "x86_64"


def test_sil_link_real_profile_only_link_real() -> None:
    """profiles/sil-link-real.toml sets link='real', leaving the others 'sim'."""
    result = load_config(_DEFAULT, _profile("sil-link-real.toml"))
    assert isinstance(result, Ok)
    env = result.value.environment
    assert env.link == "real"
    assert (env.sensor, env.gimbal, env.compute, env.clock) == (
        "sim",
        "sim",
        "sim",
        "sim",
    )


@pytest.mark.parametrize(
    "name, sensor, link, clock, host",
    [
        ("pil.toml", "sim", "real", "real", "jetson_aarch64"),
        ("hil.toml", "real", "real", "real", "jetson_aarch64"),
    ],
)
def test_defined_not_run_profiles_load(
    name: str, sensor: str, link: str, clock: str, host: str
) -> None:
    """The DEFINED-NOT-RUN profiles still load and override their axes."""
    result = load_config(_DEFAULT, _profile(name))
    assert isinstance(result, Ok)
    env = result.value.environment
    assert env.sensor == sensor
    assert env.link == link
    assert env.clock == clock
    assert env.host == host
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/flight/tests/test_profiles.py -v`
Expected: FAIL with `Err("override config not found: .../profiles/sil.toml")` -> the `isinstance(result, Ok)` assertion fails (profiles do not exist yet).

- [ ] **Step 3: Create the four profile files**

Create `profiles/sil.toml`:

```toml
# PACT deployment profile: SIL (software-in-the-loop, fully simulated, x86_64 dev/CI).
# OVERRIDE applied on top of config/default.toml via
#   load_config("config/default.toml", "profiles/sil.toml").
# Every axis is a sim stand-in; the deterministic ManualClock is selected by clock="sim".

[environment]
sensor = "sim"
gimbal = "sim"
compute = "sim"
link = "sim"
clock = "sim"
host = "x86_64"
```

Create `profiles/sil-link-real.toml`:

```toml
# PACT deployment profile: SIL with the REAL station link (real socket transport over
# loopback; all other axes simulated). Exercises CCSDS framing + TCP/UDP transport and
# the GSE StationEmulator end-to-end on x86_64, without flight camera/gimbal/compute.
# OVERRIDE applied on top of config/default.toml.

[environment]
sensor = "sim"
gimbal = "sim"
compute = "sim"
link = "real"
clock = "sim"
host = "x86_64"
```

Create `profiles/pil.toml`:

```toml
# PACT deployment profile: PIL (processor-in-the-loop) -- DEFINED, NOT RUN.
# Real compute/link/clock on the Jetson target; camera + gimbal still simulated.
# This profile is committed for completeness but is NOT executed by CI (no Jetson in CI).
# OVERRIDE applied on top of config/default.toml.

[environment]
sensor = "sim"
gimbal = "sim"
compute = "real"
link = "real"
clock = "real"
host = "jetson_aarch64"
```

Create `profiles/hil.toml`:

```toml
# PACT deployment profile: HIL (hardware-in-the-loop) -- DEFINED, NOT RUN.
# Every axis real on the Jetson target with the flight camera, gimbal, and station link.
# This profile is committed for completeness but is NOT executed by CI (no hardware in CI).
# OVERRIDE applied on top of config/default.toml.

[environment]
sensor = "real"
gimbal = "real"
compute = "real"
link = "real"
clock = "real"
host = "jetson_aarch64"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/flight/tests/test_profiles.py -v`
Expected: PASS (sil all-sim + host x86_64; sil-link-real only link real; pil/hil load with their expected axes).

- [ ] **Step 5: Run gates**

Run each and expect clean:
- `uv run ruff check packages`
- `uv run ruff format --check packages`  (TOML profiles are not Python; ruff ignores them)
- `uv run mypy packages`
- `uv run lint-imports`
- `uv run pytest packages -m "not e2e"`

- [ ] **Step 6: Commit**

```bash
git add profiles/sil.toml profiles/sil-link-real.toml profiles/pil.toml profiles/hil.toml packages/flight/tests/test_profiles.py && git commit -m "feat(profiles): add sil, sil-link-real, pil, hil environment override profiles

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

## Phase C (part 1): step_once extraction + gse scaffold + station emulator + scenario

> **SEQUENCING GATE (hard precondition for this whole Phase-C-part-1 section):**
> Tasks C2 (`gse.station`) and C4/C5 (later in Phase C) import `build_tc_packet` from
> `flight.libs.commands`. That symbol does NOT exist there yet -- as-built it lives at
> `flight.iss_iface.ingress.pipeline`, and `flight.libs.commands.__init__` currently re-exports
> only the command-dictionary symbols (`COMMAND_DICTIONARY`, `CommandSpec`, `ParamSpec`,
> `lookup_command`, `validate_command`). **Phase A (which RELOCATES `build_tc_packet` to
> `flight.libs.commands` per the shared contract) MUST land before any task in this section is
> started.** If Phase A has not landed, `from flight.libs.commands import build_tc_packet` raises
> `ImportError` and Task C2 cannot go green. Verify the relocation is present before beginning:
> `uv run python -c "from flight.libs.commands import build_tc_packet"` must exit 0. Do NOT
> work around this by importing from `flight.iss_iface.ingress` -- the contract direction
> (`flight.libs.commands`) is authoritative; only the ordering is being enforced here.
> Tasks C0, C1, and C3 do not depend on `build_tc_packet` and may proceed independently, but the
> whole section is gated on Phase A so the section can be validated end-to-end in one pass.

### Task C0: Extract `step_once` from `SilHarness.step` into `sim/sil/stepping.py`

This is a pure refactor: the body of `SilHarness.step` moves verbatim into a free,
driver-agnostic, Protocol-typed `step_once(...)` function, and `SilHarness.step`
delegates to it. The existing SIL closed-loop tests are the regression guard.

**Regression guard (existing tests -- do NOT modify them):**
`packages/sim/tests/test_sil_closed_loop.py` (all 8 tests:
`test_sil_nominal_closed_loop_tracks_plume`, `test_sil_thermal_fault_drives_safe_mode`,
`test_thermal_safe_stows_the_gimbal`, `test_safe_recovery_returns_to_operations`,
`test_tracking_commands_point_toward_the_plume`,
`test_valid_command_flows_through_to_bus_and_acks`,
`test_tampered_command_is_rejected_not_routed`).

**Files:**
- Create: `packages/sim/src/sim/sil/stepping.py`
- Modify: `packages/sim/src/sim/sil/runner.py` (`SilHarness.step`, currently lines 121-165, and imports lines 16-32)
- Modify: `packages/sim/src/sim/sil/__init__.py` (re-export `step_once`)
- Test (new, characterization for the extracted function): `packages/sim/tests/test_sil_stepping.py`

- [ ] **Step 1: Write the failing test**

Create `packages/sim/tests/test_sil_stepping.py`:

```python
"""Characterization test for the extracted driver-agnostic step_once."""

from flight.libs.config import PactConfig
from flight.libs.messages import InferenceResultMsg
from flight.libs.time import ManualClock
from flight.libs.types import Ok
from sim.scene import build_frames, plume_detector
from sim.sil import build_sil_system, step_once


def test_step_once_processes_one_frame_per_call() -> None:
    """step_once runs the full per-cycle body: one inference is published per call."""
    system = build_sil_system(
        PactConfig(),
        ManualClock(),
        build_frames(3),
        plume_detector(),
        inbound_packets=[],
        thermal_readings=[25.0, 25.0, 25.0],
        power_readings=[30.0, 30.0, 30.0],
    )
    inf_sub = system.bus.subscribe(InferenceResultMsg)
    payload_state = system.apps.payload.controller.initial_state()
    fault_entries = system.apps.fault.initial_entries()

    now = 0.0
    for _ in range(3):
        now += 1.0
        system.clock.advance(1.0)
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

    inference_count = 0
    while not inf_sub.empty():
        inf_sub.get_nowait()
        inference_count += 1
    assert inference_count == 3

    position = system.gimbal.read_position()
    assert isinstance(position, Ok)
    assert (position.value.az_deg, position.value.el_deg) != (0.0, 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/sim/tests/test_sil_stepping.py::test_step_once_processes_one_frame_per_call -v`
Expected: FAIL with `ImportError: cannot import name 'step_once' from 'sim.sil'` (module/symbol does not exist yet).

- [ ] **Step 3: Create `sim/sil/stepping.py`**

Create `packages/sim/src/sim/sil/stepping.py` (the body is the verbatim
`SilHarness.step` body from `runner.py` lines 134-165, made into a free function with
explicit Protocol-typed parameters):

```python
"""Driver-agnostic single-step body for the SIL harness and the GSE in-process backend.

step_once reproduces exactly one deterministic SIL cycle: poll mode changes, acquire +
process one payload frame (if available), sample housekeeping, pump the ISS bridge, publish
per-subsystem liveness heartbeats, then run the FDIR tick. It is Protocol-typed
(ImagingSensor / GimbalActuator / MessageBus) so both SilHarness and the GSE InProcessBackend
can reuse it without depending on concrete drivers. State (payload ControlState + the FDIR
watchdog entries) is threaded in and out, never held in this module.

Contains:
  - step_once: run one deterministic SIL cycle over the shared bus and return new state.

Satisfies: REQ-SIM-SIL-001.
"""

from __future__ import annotations

# internal
from flight.core.composition import MONITORED_SUBSYSTEMS, SystemApps
from flight.fault.watchdog import WatchdogEntry
from flight.hal.interfaces import GimbalActuator, ImagingSensor
from flight.libs.bus import MessageBus
from flight.libs.messages import HeartbeatMsg
from flight.libs.time import ManualClock
from flight.libs.types import MessageType, Ok
from flight.payload.control import ControlState


def step_once(
    apps: SystemApps,
    sensor: ImagingSensor,
    gimbal: GimbalActuator,
    bus: MessageBus,
    clock: ManualClock,
    now: float,
    payload_state: ControlState,
    fault_entries: dict[str, WatchdogEntry],
) -> tuple[ControlState, dict[str, WatchdogEntry]]:
    """Advance every subsystem one deterministic cycle over the shared bus.

    Order: poll mode changes -> acquire + process one payload frame (if available) ->
    housekeeping handle-commands + sample -> ISS bridge pump -> publish per-subsystem
    liveness heartbeats -> FDIR tick (drains heartbeats + faults, publishes any SAFE).

    Args:
        apps: The wired SystemApps (payload / fault / iss_iface / thermal / electrical).
        sensor: The imaging sensor Protocol the payload acquires a frame from this cycle.
        gimbal: The gimbal actuator Protocol whose position feeds the payload controller.
        bus: The shared in-process MessageBus all apps publish/subscribe on.
        clock: The ManualClock supplying wall-clock timestamps for the heartbeats.
        now: Monotonic seconds for the arbiter and watchdog (advanced by the caller).
        payload_state: The payload ControlState threaded in from the previous cycle.
        fault_entries: The FDIR watchdog entries threaded in from the previous cycle.

    Returns:
        A tuple of the new payload ControlState and the new FDIR watchdog entries.

    Notes:
        Driver-agnostic by construction: it imports only HAL Protocols + apps, never a
        concrete driver, so the GSE in-process backend reuses it verbatim. The body is the
        single source of truth for one SIL cycle; SilHarness.step delegates here.
    """
    safe_commanded, safe_cleared = apps.payload.poll_mode_changes()
    acquired = sensor.acquire_frame()
    if isinstance(acquired, Ok):
        pos = gimbal.read_position()
        payload_state, _ = apps.payload.process_frame(
            acquired.value,
            payload_state,
            now,
            0.0,
            pos.value if isinstance(pos, Ok) else None,
            safe_commanded,
            safe_cleared,
        )

    apps.thermal.handle_commands()
    apps.thermal.sample()
    apps.electrical.handle_commands()
    apps.electrical.sample()

    apps.iss_iface.tick()

    for subsystem in MONITORED_SUBSYSTEMS:
        bus.publish(
            HeartbeatMsg(
                msg_type=MessageType.HEARTBEAT,
                timestamp_utc=clock.wall_clock_iso(),
                subsystem=subsystem,
                sequence=0,
            )
        )

    fault_entries = apps.fault.tick(fault_entries, now)
    return payload_state, fault_entries
```

Now rewrite `SilHarness.step` in `packages/sim/src/sim/sil/runner.py` to delegate. Replace
the existing `step` method body (lines 121-165) with:

```python
    def step(self, now: float) -> None:
        """Advance every subsystem one cycle over the shared bus (delegates to step_once).

        Args:
            now: Monotonic seconds for the arbiter and watchdog (advanced by the caller).
        """
        system = self._system
        self._payload_state, self._fault_entries = step_once(
            system.apps,
            system.sensor,
            system.gimbal,
            system.bus,
            system.clock,
            now,
            self._payload_state,
            self._fault_entries,
        )
```

Add the `step_once` import to `runner.py`. After the existing internal import block
(the `from flight.payload.model import ScriptedDetector` line, line 32), add:

```python
from sim.sil.stepping import step_once
```

Remove the now-unused imports from `runner.py` that only `step` used: `HeartbeatMsg`
(line 27) and `MessageType` (line 29 -- keep `GimbalState, MosaicFrame, Ok` from that
same line, drop only `MessageType`). The line 29 import becomes:

```python
from flight.libs.types import GimbalState, MosaicFrame, Ok
```

and delete the line 27 `from flight.libs.messages import HeartbeatMsg`. (`MONITORED_SUBSYSTEMS`
is still used by `build_sil_system` so keep the line 22 import; `Ok` is still used by
`build_sil_system` callers -- confirm with ruff in Step 5 and drop `Ok` too if ruff reports it
unused.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/sim/tests/test_sil_stepping.py packages/sim/tests/test_sil_closed_loop.py -v`
Expected: PASS (the new characterization test plus all 8 existing closed-loop regression tests pass unchanged).

- [ ] **Step 5: Run gates**

```bash
uv run ruff check packages
uv run ruff format --check packages
uv run mypy packages
uv run lint-imports
uv run pytest packages -m "not e2e"
```
Expected: all green. (If ruff flags any leftover unused import in `runner.py`, remove it.)
`lint-imports` must still pass: `stepping.py` imports only `flight.core.composition`,
HAL Protocols, and `flight.libs.*` -- no concrete driver -- so `drivers-independent` and the
flight-layer contracts are unaffected (this is `sim`, not `flight`).

- [ ] **Step 6: Commit**

```bash
git add packages/sim/src/sim/sil/stepping.py packages/sim/src/sim/sil/runner.py packages/sim/src/sim/sil/__init__.py packages/sim/tests/test_sil_stepping.py
git commit -m "$(cat <<'EOF'
refactor(sil): extract driver-agnostic step_once from SilHarness.step

Move the per-cycle SIL body verbatim into sim/sil/stepping.py as a
Protocol-typed free function so the GSE in-process backend can reuse it.
SilHarness.step now delegates. Existing closed-loop tests guard the refactor.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

Also update `packages/sim/src/sim/sil/__init__.py` to re-export `step_once` (do this in
Step 3, listed here for completeness):

```python
"""SIL harness: run the real flight apps over sim drivers and step them deterministically."""

from sim.sil.runner import SilHarness, SilSystem, build_sil_system
from sim.sil.stepping import step_once

__all__ = ["SilHarness", "SilSystem", "build_sil_system", "step_once"]
```

---

### Task C1: Scaffold the `pact-gse` workspace package

Create the new `packages/gse` workspace package (layer: imports `flight.libs` + `sim` ONLY)
and register it across the root `pyproject.toml` workspace/sources/mypy/pytest/dev-extras.

**Files:**
- Create: `packages/gse/pyproject.toml`
- Create: `packages/gse/src/gse/__init__.py`
- Create: `packages/gse/src/gse/py.typed`
- Create: `packages/gse/tests/test_import.py`
- Modify: `pyproject.toml` (root): `[tool.uv.workspace] members` (line 99); `[tool.uv.sources]` (after line 104); `[tool.mypy] mypy_path` (line 87); `[tool.pytest.ini_options] testpaths` (lines 62-67); `dev` extras list (lines 35-49).

- [ ] **Step 1: Write the failing test**

Create `packages/gse/tests/test_import.py`:

```python
"""Smoke test: the gse package is importable and declares py.typed."""

import gse


def test_gse_imports() -> None:
    """Importing gse succeeds and exposes its dunder version."""
    assert gse.__name__ == "gse"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/gse/tests/test_import.py -v`
Expected: FAIL -- collection error `ModuleNotFoundError: No module named 'gse'` (package not created / not registered in the workspace yet).

- [ ] **Step 3: Create the package and register it**

Create `packages/gse/pyproject.toml`:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "pact-gse"
version = "0.1.0"
description = "PACT ground support equipment: station emulator + scenario harness (test tooling)"
requires-python = ">=3.14"
dependencies = [
    "numpy>=1.24",
    "pact-flight",
    "pact-sim",
]

[tool.hatch.build.targets.wheel]
packages = ["src/gse"]

[tool.uv.sources]
pact-flight = { workspace = true }
pact-sim = { workspace = true }
```

Create `packages/gse/src/gse/__init__.py`:

```python
"""PACT ground support equipment (GSE): station emulator + deterministic scenario harness.

GSE is OUT-OF-FLIGHT test tooling. It stands in for the real ISS ground segment so the flight
software's command-ingress and downlink paths can be exercised end-to-end (sockets for PIL/HIL,
in-process for SIL). gse depends ONLY on flight.libs (CCSDS framing, the command dictionary,
build_tc_packet) and sim (scene + step_once); flight and sim must never import gse (enforced by
the flight-gse-isolation / sim-gse-isolation import-linter contracts).

Satisfies: REQ-VAL-GSE-001.
"""
```

Create `packages/gse/src/gse/py.typed` (empty marker file):

```
```

Modify root `pyproject.toml`. Update line 99 `[tool.uv.workspace] members`:

```toml
members = ["packages/flight", "packages/sim", "packages/tools", "packages/gse"]
```

Add to `[tool.uv.sources]` (after the `pact-tools` line, line 104):

```toml
pact-gse = { workspace = true }
```

Update `[tool.mypy] mypy_path` (line 87):

```toml
mypy_path = ["packages/flight/src", "packages/sim/src", "packages/tools/src", "packages/gse/src"]
```

Update `[tool.pytest.ini_options] testpaths` (lines 62-67) to add the gse tests dir:

```toml
testpaths = [
    "tests",
    "packages/flight/tests",
    "packages/sim/tests",
    "packages/tools/tests",
    "packages/gse/tests",
]
```

Add `"pact-gse"` to the `dev` extras editable list (the block ending line 49), after
`"pact-tools",`:

```toml
    "pact-flight",
    "pact-sim",
    "pact-tools",
    "pact-gse",
```

Then sync so the new editable member is installed:

Run: `uv sync --extra dev`
Expected: resolves and installs `pact-gse` editable (and re-installs the other members).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/gse/tests/test_import.py -v`
Expected: PASS.

- [ ] **Step 5: Run gates**

```bash
uv run ruff check packages
uv run ruff format --check packages
uv run mypy packages
uv run lint-imports
uv run pytest packages -m "not e2e"
```
Expected: all green. `lint-imports` still passes (no gse-isolation contracts added yet --
those land in Phase C part 2; the `__init__.py` imports nothing from flight/sim so there is
nothing to violate now).

- [ ] **Step 6: Commit**

```bash
git add packages/gse pyproject.toml
git commit -m "$(cat <<'EOF'
build(gse): scaffold pact-gse workspace package

New out-of-flight GSE package (deps: numpy, pact-flight, pact-sim).
Registered in the uv workspace, sources, mypy_path, pytest testpaths,
and the dev editable extras. Empty package + import smoke test.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task C2: `gse.station.StationEmulator`

A test-side ground station: a TCP client that connects to the flight `RealStationLink`'s
inbound TC server, signs commands with `flight.libs.commands.build_tc_packet`, and a UDP
receiver bound on the telemetry endpoint to drain downlinked TM. As GSE test tooling
(not flight library code) it may raise on misuse rather than returning `Result`.

> **HARD DEPENDENCY -- Phase A MUST have landed.** This task imports `build_tc_packet` from
> `flight.libs.commands`. The shared contract has Phase A RELOCATE `build_tc_packet` from its
> as-built home (`flight.iss_iface.ingress.pipeline`) to `flight.libs.commands`. Until that
> relocation is merged, `from flight.libs.commands import build_tc_packet` raises `ImportError`
> and this task CANNOT go green. Before starting, confirm the relocation is present:
> `uv run python -c "from flight.libs.commands import build_tc_packet"` must exit 0. Import it
> from `flight.libs.commands` per the contract -- do NOT fall back to importing from
> `flight.iss_iface.ingress`.

**Files:**
- Create: `packages/gse/src/gse/station.py`
- Test: `packages/gse/tests/test_station_emulator.py`

- [ ] **Step 1: Write the failing test**

Create `packages/gse/tests/test_station_emulator.py`:

```python
"""StationEmulator round-trip against a real flight RealStationLink over loopback sockets."""

import dataclasses
import socket
import time

from flight.hal.drivers_real.station import RealStationLink
from flight.libs.config import LinkConfig
from flight.libs.time import RealClock
from flight.libs.types import Ok
from gse.station import StationEmulator

_KEY = b"sil-test-key-0000000000000000000"


def _free_port() -> int:
    """Return an OS-assigned free TCP port (released immediately; race window is tiny)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_station_emulator_round_trip() -> None:
    """A signed command reaches the flight link; a TM datagram reaches the emulator."""
    tc_port = _free_port()
    udp_port = _free_port()
    cfg = dataclasses.replace(
        LinkConfig(),
        command_tcp_host="127.0.0.1",
        command_tcp_port=tc_port,
        telemetry_udp_host="127.0.0.1",
        telemetry_udp_port=udp_port,
    )
    link = RealStationLink(cfg, RealClock())
    emulator = StationEmulator(
        tcp_host="127.0.0.1",
        tcp_port=tc_port,
        udp_host="127.0.0.1",
        udp_port=udp_port,
        key=_KEY,
        tc_apid=cfg.tc_apid,
    )
    try:
        emulator.connect()
        emulator.send_command("SET_THERMAL_LIMIT", {"limit_c": 70.0}, "ground", 1)

        received: bytes | None = None
        for _ in range(50):  # allow the daemon recv/deframe thread to land the packet
            popped = link.receive_packet()
            assert isinstance(popped, Ok)
            if popped.value is not None:
                received = popped.value
                break
            time.sleep(0.02)
        assert received is not None
        assert len(received) > 0

        sent = link.send_packet(b"<tm-downlink>")
        assert isinstance(sent, Ok)
        drained = emulator.poll_downlink(timeout_s=0.5)
        assert b"<tm-downlink>" in drained
    finally:
        emulator.close()
        link.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/gse/tests/test_station_emulator.py::test_station_emulator_round_trip -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gse.station'` (module not created yet).
(If you instead see `ImportError: cannot import name 'build_tc_packet' from 'flight.libs.commands'`,
Phase A has NOT landed -- STOP and resolve the sequencing gate above before continuing.)

- [ ] **Step 3: Create `gse/station.py`**

Create `packages/gse/src/gse/station.py`:

```python
"""Ground-station emulator: TCP command client + UDP telemetry receiver for the flight link.

StationEmulator is the test-side counterpart to the flight RealStationLink. RealStationLink
binds a TCP SERVER for inbound telecommands and SENDS UDP telemetry; so the emulator connects
a TCP CLIENT to that server (to push signed TC packets) and binds a UDP RECEIVER on the
telemetry endpoint (to drain downlinked TM). Commands are framed + signed by the flight
build_tc_packet so the flight ingress pipeline authenticates them identically to the real
ground segment. This is GSE test tooling (not flight library code): methods raise on misuse
rather than returning Result.

Contains:
  - StationEmulator: connect / send_command / poll_downlink / close.

Satisfies: REQ-VAL-GSE-001.
"""

from __future__ import annotations

# stdlib
import socket

# internal
from flight.libs.commands import build_tc_packet


class StationEmulator:
    """Emulated ISS ground station: signs + sends telecommands, receives telemetry datagrams."""

    def __init__(
        self,
        tcp_host: str,
        tcp_port: int,
        udp_host: str,
        udp_port: int,
        key: bytes,
        tc_apid: int,
    ) -> None:
        """Record the flight link endpoints and the shared HMAC key (no sockets opened yet).

        Args:
            tcp_host: Host of the flight link's inbound TC server (the emulator connects here).
            tcp_port: TCP port of the flight link's inbound TC server.
            udp_host: Host the emulator binds to receive outbound telemetry datagrams.
            udp_port: UDP port the emulator binds to receive outbound telemetry datagrams.
            key: The shared HMAC-SHA256 secret used to sign telecommands.
            tc_apid: The CCSDS APID stamped into outbound telecommand packets.

        Notes:
            Sockets are opened in connect(), not here, so an unconnected emulator is inert.
        """
        self._tcp_host = tcp_host
        self._tcp_port = tcp_port
        self._udp_host = udp_host
        self._udp_port = udp_port
        self._key = key
        self._tc_apid = tc_apid
        self._tcp: socket.socket | None = None
        self._udp: socket.socket | None = None

    def connect(self) -> None:
        """Open the TCP client to the flight TC server and bind the UDP telemetry receiver.

        Raises:
            OSError: if the TCP connect or the UDP bind fails (test-setup error).
            RuntimeError: if called when already connected.
        """
        if self._tcp is not None or self._udp is not None:
            raise RuntimeError("StationEmulator is already connected")
        udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        udp.bind((self._udp_host, self._udp_port))
        udp.setblocking(False)
        self._udp = udp
        tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp.connect((self._tcp_host, self._tcp_port))
        self._tcp = tcp

    def send_command(
        self,
        command_id: str,
        params: dict[str, str | int | float | bool],
        source: str,
        seq: int,
    ) -> None:
        """Frame, sign, and transmit one telecommand over the TCP connection to the flight link.

        Args:
            command_id: The command opcode string (e.g. "SET_THERMAL_LIMIT", "PING").
            params: The command parameter dict.
            source: The command origin identifier (must be on the flight allow-list to accept).
            seq: The per-source monotonic sequence number.

        Returns:
            None.

        Raises:
            RuntimeError: if called before connect().
            ValueError: if build_tc_packet rejects a field (propagated from the flight builder).
        """
        if self._tcp is None:
            raise RuntimeError("StationEmulator.connect() must be called before send_command()")
        packet = build_tc_packet(command_id, params, source, seq, self._key, self._tc_apid)
        self._tcp.sendall(packet)

    def poll_downlink(self, timeout_s: float = 0.5) -> list[bytes]:
        """Drain all telemetry datagrams currently waiting on the UDP receiver.

        Args:
            timeout_s: Total wall-clock budget to wait for at least one datagram (seconds).

        Returns:
            A list of received datagram payloads (may be empty if none arrived in the budget).

        Raises:
            RuntimeError: if called before connect().

        Notes:
            Blocks up to timeout_s for the first datagram, then drains any others non-blocking,
            so a TM sent just before the call is reliably captured without busy-spinning.
        """
        if self._udp is None:
            raise RuntimeError("StationEmulator.connect() must be called before poll_downlink()")
        self._udp.settimeout(timeout_s)
        received: list[bytes] = []
        try:
            received.append(self._udp.recv(65535))
        except (TimeoutError, BlockingIOError, OSError):
            return received
        self._udp.setblocking(False)
        while True:
            try:
                received.append(self._udp.recv(65535))
            except (BlockingIOError, OSError):
                break
        return received

    def close(self) -> None:
        """Close both sockets (idempotent; safe to call without a prior connect())."""
        for sock in (self._tcp, self._udp):
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
        self._tcp = None
        self._udp = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/gse/tests/test_station_emulator.py::test_station_emulator_round_trip -v`
Expected: PASS (signed command deframed by the flight link's daemon thread within the retry
loop; the TM datagram drained by `poll_downlink`).

- [ ] **Step 5: Run gates**

```bash
uv run ruff check packages
uv run ruff format --check packages
uv run mypy packages
uv run lint-imports
uv run pytest packages -m "not e2e"
```
Expected: all green. `gse.station` imports only `flight.libs.commands` + stdlib `socket`,
satisfying the gse layer rule (flight.libs + sim only).

- [ ] **Step 6: Commit**

```bash
git add packages/gse/src/gse/station.py packages/gse/tests/test_station_emulator.py
git commit -m "$(cat <<'EOF'
feat(gse): StationEmulator (TCP command client + UDP telemetry receiver)

Emulated ground station that signs telecommands via flight build_tc_packet
and round-trips against RealStationLink over loopback. send_command pushes
TC over TCP; poll_downlink drains TM over UDP.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task C3: `gse.scenario` dataclasses + `load_scenario`

Frozen scenario dataclasses (`Scenario`, `SceneSpec`, `CommandStep`, `Assertion`) and a
`tomllib`-backed `load_scenario(path)`. Assertions carry a `tag` distinguishing
`frame-portable` (scored under the deterministic in-process backend) from `realtime-only`.

**Files:**
- Create: `packages/gse/src/gse/scenario.py`
- Test: `packages/gse/tests/test_scenario.py`

- [ ] **Step 1: Write the failing test**

Create `packages/gse/tests/test_scenario.py`:

```python
"""load_scenario parses a scenario TOML into the typed dataclasses with assertion tags."""

from pathlib import Path

from gse.scenario import Assertion, CommandStep, Scenario, SceneSpec, load_scenario

_SAMPLE = """\
name = "thermal_safe"
profile = "sil"
steps = 6
dt = 1.0

[scene]
num_frames = 6
seed = 0

[[commands]]
at_frame = 1
command_id = "SET_THERMAL_LIMIT"
source = "ground"
seq = 1
params = { limit_c = 70.0 }

[[assertions]]
id = "mode_goes_safe"
kind = "mode_is"
value = "SAFE"
tag = "frame-portable"

[[assertions]]
id = "ack_is_fast"
kind = "ack_within_seconds"
value = 2.0
tag = "realtime-only"
"""


def test_load_scenario_parses_all_fields(tmp_path: Path) -> None:
    """A sample scenario TOML round-trips into Scenario with both assertion tags preserved."""
    path = tmp_path / "thermal_safe.toml"
    path.write_text(_SAMPLE, encoding="ascii")

    scenario = load_scenario(str(path))

    assert isinstance(scenario, Scenario)
    assert scenario.name == "thermal_safe"
    assert scenario.profile == "sil"
    assert scenario.steps == 6
    assert scenario.dt == 1.0

    assert scenario.scene == SceneSpec(num_frames=6, seed=0)

    assert scenario.commands == (
        CommandStep(
            at_frame=1,
            command_id="SET_THERMAL_LIMIT",
            params={"limit_c": 70.0},
            source="ground",
            seq=1,
        ),
    )

    assert len(scenario.assertions) == 2
    assert scenario.assertions[0] == Assertion(
        id="mode_goes_safe", kind="mode_is", value="SAFE", tag="frame-portable"
    )
    assert scenario.assertions[1].tag == "realtime-only"
    assert scenario.assertions[1].kind == "ack_within_seconds"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/gse/tests/test_scenario.py::test_load_scenario_parses_all_fields -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gse.scenario'` (module not created yet).

- [ ] **Step 3: Create `gse/scenario.py`**

Create `packages/gse/src/gse/scenario.py`:

```python
"""Scenario model + loader for the GSE deterministic harness.

A Scenario is a fully declarative test case: which profile to wire, what scene to render,
which commands to inject at which frame, and which assertions to score. Assertions carry a
tag: "frame-portable" assertions hold under the deterministic in-process backend (mode,
ack status, gimbal motion, counts) and are scored; "realtime-only" assertions (e.g. wall-clock
ack latency) are DEFINED here but recorded skipped-with-reason under the in-process backend.
Scenarios are loaded from TOML via tomllib (stdlib). All dataclasses are frozen.

Contains:
  - SceneSpec: which scene to render (num_frames, seed).
  - CommandStep: one command to inject at a given frame index.
  - Assertion: one scored/skipped check (id, kind, value, frame-portable|realtime-only tag).
  - Scenario: the whole declarative case.
  - load_scenario: parse a scenario TOML file into a Scenario.

Satisfies: REQ-VAL-GSE-001.
"""

from __future__ import annotations

# stdlib
import tomllib
from dataclasses import dataclass
from typing import Literal

ParamValue = str | int | float | bool
AssertionTag = Literal["frame-portable", "realtime-only"]


@dataclass(frozen=True, slots=True)
class SceneSpec:
    """Which deterministic scene the harness renders for a scenario.

    Fields:
        num_frames: Number of mosaic frames to render (one per SIL step).
        seed: Deterministic render seed.
    """

    num_frames: int
    seed: int


@dataclass(frozen=True, slots=True)
class CommandStep:
    """One telecommand to inject at a given frame index during a scenario run.

    Fields:
        at_frame: 1-based step index at which the command is injected.
        command_id: The command opcode string (e.g. "SET_THERMAL_LIMIT", "PING").
        params: The command parameter dict.
        source: The command origin identifier (must be on the flight allow-list to accept).
        seq: The per-source monotonic sequence number.
    """

    at_frame: int
    command_id: str
    params: dict[str, ParamValue]
    source: str
    seq: int


@dataclass(frozen=True, slots=True)
class Assertion:
    """One scenario assertion, scored or skipped depending on its tag.

    Fields:
        id: Stable identifier for the assertion (cited as evidence in the VCRM).
        kind: The assertion kind ("mode_is", "command_acked", "gimbal_moved",
            "min_inference_count", "min_downlink_count", "ack_within_seconds").
        value: The expected value (kind-dependent: a mode/status string, a bool, an int,
            or a float seconds budget).
        tag: "frame-portable" (scored under the in-process backend) or "realtime-only"
            (recorded skipped-with-reason under the in-process backend).
    """

    id: str
    kind: str
    value: ParamValue
    tag: AssertionTag


@dataclass(frozen=True, slots=True)
class Scenario:
    """A fully declarative GSE test case: profile + scene + commands + assertions.

    Fields:
        name: Human-readable scenario name (also the evidence id stem in the VCRM).
        profile: Profile name applied as a load_config override (e.g. "sil", "sil-link-real").
        scene: The SceneSpec to render.
        commands: The telecommands to inject, in declaration order.
        assertions: The assertions to score/skip, in declaration order.
        steps: Number of deterministic steps to run.
        dt: Seconds to advance per step.
    """

    name: str
    profile: str
    scene: SceneSpec
    commands: tuple[CommandStep, ...]
    assertions: tuple[Assertion, ...]
    steps: int
    dt: float


def load_scenario(path: str) -> Scenario:
    """Parse a scenario TOML file into a typed, frozen Scenario.

    Args:
        path: Filesystem path to the scenario TOML file.

    Returns:
        The parsed Scenario.

    Raises:
        OSError: if the file cannot be read.
        tomllib.TOMLDecodeError: if the file is not valid TOML.
        KeyError: if a required scenario/scene/command/assertion field is missing.

    Notes:
        GSE test tooling, so this raises on malformed input rather than returning a Result.
        commands/assertions are normalized to tuples so the returned Scenario is fully frozen
        and hashable. Each assertion's tag is taken verbatim from the TOML ("frame-portable"
        or "realtime-only") and is the only signal the orchestrator uses to score-vs-skip it.
    """
    with open(path, "rb") as handle:
        data = tomllib.load(handle)

    scene_raw = data["scene"]
    scene = SceneSpec(num_frames=int(scene_raw["num_frames"]), seed=int(scene_raw["seed"]))

    commands = tuple(
        CommandStep(
            at_frame=int(cmd["at_frame"]),
            command_id=str(cmd["command_id"]),
            params=dict(cmd.get("params", {})),
            source=str(cmd["source"]),
            seq=int(cmd["seq"]),
        )
        for cmd in data.get("commands", [])
    )

    assertions = tuple(
        Assertion(
            id=str(item["id"]),
            kind=str(item["kind"]),
            value=item["value"],
            tag=_parse_tag(item["tag"]),
        )
        for item in data.get("assertions", [])
    )

    return Scenario(
        name=str(data["name"]),
        profile=str(data["profile"]),
        scene=scene,
        commands=commands,
        assertions=assertions,
        steps=int(data["steps"]),
        dt=float(data["dt"]),
    )


def _parse_tag(raw: object) -> AssertionTag:
    """Validate a raw TOML tag string against the allowed assertion tags.

    Args:
        raw: The tag value read from the TOML assertion table.

    Returns:
        The validated AssertionTag literal.

    Raises:
        ValueError: if the tag is not "frame-portable" or "realtime-only".
    """
    if raw == "frame-portable":
        return "frame-portable"
    if raw == "realtime-only":
        return "realtime-only"
    raise ValueError(f"unknown assertion tag: {raw!r}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/gse/tests/test_scenario.py::test_load_scenario_parses_all_fields -v`
Expected: PASS.

- [ ] **Step 5: Run gates**

```bash
uv run ruff check packages
uv run ruff format --check packages
uv run mypy packages
uv run lint-imports
uv run pytest packages -m "not e2e"
```
Expected: all green. `gse.scenario` imports only stdlib (`tomllib`, `dataclasses`, `typing`) --
no flight/sim import -- so no layer concern.

- [ ] **Step 6: Commit**

```bash
git add packages/gse/src/gse/scenario.py packages/gse/tests/test_scenario.py
git commit -m "$(cat <<'EOF'
feat(gse): scenario model + tomllib load_scenario

Frozen Scenario/SceneSpec/CommandStep/Assertion dataclasses plus a stdlib
load_scenario. Assertions carry a frame-portable|realtime-only tag that the
orchestrator uses to decide score-vs-skip under the in-process backend.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

## Phase C (part 2): harness backend + orchestrator + sil-link-real integration

> **Phase ordering (HARD CONSTRAINT — read before starting any task in this section).**
> Every task in this section (C4, C5, C6) is downstream of work in **Phase A**, **Phase B**, and **Phase C (part 1)**. In particular:
> - **Phase A MUST land before C4/C6.** C4 (`gse/harness.py`) imports `build_tc_packet` from `flight.libs.commands`, and C2's `StationEmulator` + C3's scenario tooling do the same. Phase A is the task that RELOCATES `build_tc_packet` into `flight.libs.commands` (the as-built `flight.libs.commands.__init__` exports only the command-dictionary symbols). If any C-phase task runs before Phase A, every `gse` import fails with `ImportError: cannot import name 'build_tc_packet'`. Phase A also fixes the upstream SyntaxError blocker, so sequencing A first is mandatory regardless.
> - **Phase B MUST land before C4/C6.** C4 calls `flight.core.select_drivers.select_drivers` + `SimDriverInputs` (built in Phase B) and loads `profiles/sil.toml` / `profiles/sil-link-real.toml` (created in Phase B).
> - **Phase C part 1 (C1-C3) MUST land before C4.** C4 imports `sim.sil.stepping.step_once` (C1), `gse.station.StationEmulator` (C2), and the `gse.scenario` types (C3).
>
> The orchestrator that assembles these sections MUST enforce the order **A -> B -> C1/C2/C3 -> C4 -> C5 -> C6**. Do not attempt C4 against an unrelocated `build_tc_packet`.
>
> **`gse` dependency-surface note (documented, contract-legal).** The SHARED CONTRACT summarizes the `gse` layer as "imports `flight.libs` + `sim` ONLY", but `InProcessBackend` necessarily also imports the `flight.core` composition-root *seams* (`load_config`, `build_apps`, `select_drivers`, `MONITORED_SUBSYSTEMS`) plus the Protocol/state types it must thread through `step_once` (`flight.fault.watchdog.WatchdogEntry`, `flight.payload.control.ControlState`, `flight.payload.calibration_io.build_identity_calibration`). This is unavoidable: `step_once` itself already depends on `flight.core.composition` + `flight.fault.watchdog`, so full avoidance is impossible. The import-linter contracts added in Phase D (`flight-gse-isolation`, `sim-gse-isolation`) only enforce the ONE-WAY rule (`flight !-> gse`, `sim !-> gse`); they do not forbid `gse -> flight.core/payload/fault`, so `lint-imports` stays green. This widened-but-legal surface is a **conscious decision**, not an accident: Phase D documents it in `packages/gse/src/gse/CONTEXT.md` and amends the SHARED CONTRACT prose to read "`gse` imports `flight.libs` + `sim` + the `flight.core` composition-root seams (`load_config`, `build_apps`, `select_drivers`) and the Protocol/state types threaded through `step_once`". The tasks below depend on that amendment being in place.

### Task C4: gse.harness — HarnessBackend Protocol + InProcessBackend + SocketBackend

**Files:**
- Create: `packages/gse/src/gse/harness.py`
- Create: `packages/gse/tests/test_harness_inprocess.py`

This task builds the `gse.harness` module on top of the already-built `gse.station.StationEmulator` (Task C2), `gse.scenario` types (Task C3), `flight.core.select_drivers` (Phase B), `sim.sil.stepping.step_once` (Task C1), and `flight.core.config_loader.load_config`. The `InProcessBackend` is the deterministic, frame-portable backend; `SocketBackend` is a declared-but-deferred stub.

Driver-axis selection lives entirely in `flight.core.select_drivers.select_drivers`; the backend never imports `drivers_real`/`drivers_sim` (the composition-root seam keeps `gse` driver-clean). The backend picks `ManualClock` (the clock axis is decided by the composition root, never by `select_drivers`).

**`gse` surface (per the section header).** `InProcessBackend` imports `flight.libs`, `sim`, the `flight.core` composition-root seams (`load_config`, `build_apps`, `select_drivers`, `MONITORED_SUBSYSTEMS`), and the Protocol/state types threaded through `step_once` (`WatchdogEntry`, `ControlState`, `build_identity_calibration`). This is the documented, contract-legal widened surface — `lint-imports` enforces only the one-way `flight/sim !-> gse` rule.

For the **sim command path** there is a structural limitation in two parts:
1. `SimStationLink`'s inbound queue is fixed at construction time (it takes `inbound: list[bytes]` once). So all scenario commands destined for a sim link must be pre-baked into `SimDriverInputs.inbound_packets` at `build()` time from the scenario's command timeline. `inject_command()` on the sim path is therefore a documented no-op (the bytes are already queued).
2. Because `IssIfaceApp.pump_uplink` drains **every** available inbound packet on the first `tick()`, a sim-link scenario does **not** honor `CommandStep.at_frame` timing — every pre-baked command is ingested on the first step regardless of its `at_frame`. **Scenario-authoring constraint:** frame-portable sim-link scenarios MUST be insensitive to command-arrival ordering (assert end-state, not per-frame timing). Timed delivery is only available on the **real link path** (C6), where the `StationEmulator` uplinks each command live over TCP at its `at_frame`. This limitation is documented on the `InProcessBackend.build` docstring below.

- [ ] **Step 1: Write the failing test**

```python
"""InProcessBackend over the all-sim profile: build, step, collect a capture."""

from gse.harness import InProcessBackend, SocketBackend
from gse.scenario import Scenario, SceneSpec


def _sil_scenario() -> Scenario:
    """A minimal all-sim scenario: a short plume scene, no commands, no assertions."""
    return Scenario(
        name="harness-smoke",
        profile="profiles/sil.toml",
        scene=SceneSpec(num_frames=4, seed=0),
        commands=(),
        assertions=(),
        steps=4,
        dt=1.0,
    )


def test_inprocess_backend_builds_steps_and_collects() -> None:
    """Building over profiles/sil.toml then stepping yields a capture with inference results."""
    backend = InProcessBackend()
    backend.build(_sil_scenario(), "profiles/sil.toml")
    for i in range(4):
        backend.step(float(i + 1))
    capture = backend.collect()
    backend.shutdown()

    # One inference per stepped frame; no SAFE mode change in the nominal scene.
    assert capture.inference_count == 4
    assert capture.mode_changes == ()
    # The closed loop tracked the off-center plume and moved the gimbal off the origin
    # (off-origin past the 0.1 deg encoder-noise tolerance).
    assert capture.gimbal_moved is True


def test_socket_backend_is_deferred() -> None:
    """SocketBackend is declared but not implemented (PIL/HIL deferred)."""
    backend = SocketBackend()
    try:
        backend.build(_sil_scenario(), "profiles/pil.toml")
    except NotImplementedError as exc:
        assert "deferred" in str(exc)
    else:  # pragma: no cover - guard
        raise AssertionError("SocketBackend.build must raise NotImplementedError")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/gse/tests/test_harness_inprocess.py -v`  Expected: FAIL with `ModuleNotFoundError: No module named 'gse.harness'` (the module does not exist yet).

- [ ] **Step 3: Implement `gse/harness.py`**

```python
"""GSE harness backends: deterministic in-process vs. deferred socket transport.

A HarnessBackend abstracts how a scenario is built, stepped, command-injected, and
captured -- so the orchestrator scores the same assertions regardless of transport.

InProcessBackend is the blessed x86 backend. It builds a PactConfig from the profile
(load_config(default, profile)), renders the plume scene (sim.scene), picks a
ManualClock, selects drivers via flight.core.select_drivers (NEVER importing concrete
driver modules -- the composition-root seam keeps gse driver-clean), wires the apps via
build_apps, and steps them single-threaded through sim.sil.stepping.step_once. When the
profile's link axis is "real" it stands up a RealStationLink (chosen by select_drivers
over a free TCP/UDP port pair) and a GSE StationEmulator as the live counterpart, so the
authenticated command + CCSDS downlink path runs over real sockets in one process.

For an all-sim link, SimStationLink's inbound queue is fixed at construction, so all
scenario commands are pre-baked into SimDriverInputs.inbound_packets at build() time from
the command timeline; inject_command() is then a documented no-op on that path. Because
IssIfaceApp.pump_uplink drains every queued packet on the first tick, sim-link scenarios
do NOT honor CommandStep.at_frame timing (all commands land on step 1); author sim-link
scenarios to be insensitive to command-arrival ordering. Timed delivery is real-link only.

SocketBackend is declared (PIL/HIL transport) but raises NotImplementedError -- those
venues are DEFINED, NOT RUN.

Dependency surface: this module imports flight.libs, sim, the flight.core composition-root
seams (load_config, build_apps, select_drivers, MONITORED_SUBSYSTEMS), and the
Protocol/state types threaded through step_once (WatchdogEntry, ControlState,
build_identity_calibration). lint-imports enforces only the one-way flight/sim !-> gse rule.

Contains:
  - TelemetryCapture: frozen holder of scored bus events + downlink bytes.
  - HarnessBackend: the runtime-checkable backend Protocol.
  - InProcessBackend: deterministic ManualClock + step_once backend (sim or real link).
  - SocketBackend: deferred PIL/HIL socket backend (NotImplementedError).

Satisfies: REQ-COMM-HIGH-001, REQ-COMM-HIGH-003, REQ-GIMB-HIGH-001.
"""

from __future__ import annotations

# stdlib
import socket
from dataclasses import dataclass, replace
from typing import Protocol, TypeVar, runtime_checkable

# internal
from flight.core.composition import MONITORED_SUBSYSTEMS, build_apps
from flight.core.config_loader import load_config
from flight.core.select_drivers import SimDriverInputs, select_drivers
from flight.fault.watchdog import WatchdogEntry
from flight.libs.bus import MessageBus, Subscription
from flight.libs.commands import build_tc_packet
from flight.libs.config import LinkConfig, PactConfig
from flight.libs.messages import (
    CommandAckMsg,
    GimbalCommandMsg,
    InferenceResultMsg,
    ModeChangeMsg,
)
from flight.libs.time import ManualClock
from flight.libs.types import AckStatus, Err, GimbalState, SystemMode
from flight.payload.calibration_io import build_identity_calibration
from flight.payload.control import ControlState
from gse.scenario import CommandStep, Scenario
from gse.station import StationEmulator
from sim.scene import build_frames, plume_detector
from sim.sil.stepping import step_once

_SIL_KEY = b"sil-test-key-0000000000000000000"

# Off-origin tolerance (deg) for the gimbal-moved flag. SimGimbal.read_position() adds
# encoder noise at config.gimbal.sim_encoder_noise_deg (default 0.005 deg 1-sigma) on every
# read, so a strict != (0.0, 0.0) test would spuriously report motion even when stationary.
# 0.1 deg is 20x the noise 1-sigma, well below real tracked motion (degrees).
_GIMBAL_MOVED_TOLERANCE_DEG = 0.1

_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class TelemetryCapture:
    """Scored telemetry collected from one stepped scenario run.

    Fields:
        inference_count: Number of InferenceResultMsg published over the run.
        gimbal_moved: True if the payload moved the gimbal off the (0, 0) origin by more
            than the encoder-noise tolerance (_GIMBAL_MOVED_TOLERANCE_DEG).
        mode_changes: The SystemMode of every ModeChangeMsg, in publication order.
        acks: The AckStatus of every CommandAckMsg observed on the bus, in order.
        downlink_packets: Raw CCSDS TM datagrams the StationEmulator received over UDP
            (empty for an all-sim link, where no real socket carries downlink).
    """

    inference_count: int
    gimbal_moved: bool
    mode_changes: tuple[SystemMode, ...]
    acks: tuple[AckStatus, ...]
    downlink_packets: tuple[bytes, ...]


@runtime_checkable
class HarnessBackend(Protocol):
    """Transport-agnostic scenario backend the orchestrator drives.

    Implementations build a scenario over a profile, step it, inject commands, collect a
    TelemetryCapture for scoring, and shut down. The orchestrator scores the same
    frame-portable assertions against the capture regardless of which backend produced it.
    """

    def build(self, scenario: Scenario, profile_path: str) -> None:
        """Construct the system for scenario under the config override at profile_path."""
        ...

    def step(self, now: float) -> None:
        """Advance the system one deterministic cycle at monotonic-seconds now."""
        ...

    def inject_command(self, step: CommandStep) -> None:
        """Deliver one scenario command to the system (live over the link, or a no-op)."""
        ...

    def collect(self) -> TelemetryCapture:
        """Drain accumulated telemetry into a frozen TelemetryCapture for scoring."""
        ...

    def shutdown(self) -> None:
        """Release any sockets/threads the backend stood up (idempotent)."""
        ...


def _free_port_pair() -> tuple[int, int]:
    """Reserve two distinct free localhost TCP/UDP ports by transient binds.

    Returns:
        tuple[int, int]: A (tcp_port, udp_port) pair that were free at probe time.

    Notes:
        Binds two ephemeral sockets, reads the OS-assigned ports, then closes them. A
        narrow TOCTOU window remains, but localhost in a single CI job makes a collision
        practically impossible; the alternative (passing live sockets) breaks the
        RealStationLink/StationEmulator contract, which both take host+port.
    """
    probes: list[socket.socket] = []
    ports: list[int] = []
    for _ in range(2):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))
        ports.append(s.getsockname()[1])
        probes.append(s)
    for s in probes:
        s.close()
    return ports[0], ports[1]


def _scenario_packets(scenario: Scenario, key: bytes, tc_apid: int) -> list[bytes]:
    """Build signed TC packets for every command step in a scenario (sim-link pre-bake).

    Args:
        scenario: The scenario whose command timeline to serialize.
        key: The shared HMAC-SHA256 uplink secret.
        tc_apid: The telecommand APID from the resolved LinkConfig.

    Returns:
        list[bytes]: One framed CCSDS TC packet per CommandStep, in timeline order. These
        seed SimStationLink.inbound at build time because its queue is fixed at
        construction (the sim path cannot accept a live mid-run injection). Note that
        pump_uplink drains all of them on the first tick, so at_frame timing is NOT honored
        on the sim link.
    """
    return [
        build_tc_packet(step.command_id, step.params, step.source, step.seq, key, tc_apid)
        for step in scenario.commands
    ]


class InProcessBackend:
    """Deterministic single-process backend: ManualClock + step_once over selected drivers."""

    def __init__(self) -> None:
        """Initialize an unbuilt backend; build() populates the system state."""
        self._config: PactConfig | None = None
        self._apps_built = False
        self._bus = MessageBus()
        self._clock = ManualClock()
        self._sensor: object | None = None
        self._gimbal: object | None = None
        self._apps: object | None = None
        self._payload_state: ControlState | None = None
        self._fault_entries: dict[str, WatchdogEntry] = {}
        self._emulator: StationEmulator | None = None
        self._inf_sub: Subscription[InferenceResultMsg] | None = None
        self._gimbal_sub: Subscription[GimbalCommandMsg] | None = None
        self._mode_sub: Subscription[ModeChangeMsg] | None = None
        self._ack_sub: Subscription[CommandAckMsg] | None = None
        self._link_real = False

    def build(self, scenario: Scenario, profile_path: str) -> None:
        """Build the wired apps + drivers for scenario under the profile override.

        Args:
            scenario: The scenario (scene spec + command timeline) to realize.
            profile_path: Path to the profile TOML applied as an override over
                config/default.toml (selects the per-axis sim/real environment).

        Notes:
            For a real link axis, frees a TCP/UDP port pair, replaces LinkConfig so
            RealStationLink and StationEmulator agree on the endpoints, runs select_drivers
            (which builds RealStationLink over those ports), and connects a StationEmulator
            as the live counterpart so AOS holds and downlink datagrams are captured. For a
            sim link, the command timeline is pre-baked into the SimStationLink inbound
            queue (its queue is fixed at construction). Because pump_uplink drains every
            queued packet on the first tick, sim-link CommandStep.at_frame timing is NOT
            honored (all commands ingest on step 1); sim-link scenarios must be
            order-insensitive. Subscriptions are created BEFORE any step so no published
            message is missed (the bus only delivers to live subs).
        """
        loaded = load_config("config/default.toml", profile_path)
        if isinstance(loaded, Err):
            raise ValueError(f"profile load failed: {loaded.error}")
        config = loaded.value
        self._link_real = config.environment.link == "real"

        frames = build_frames(scenario.scene.num_frames, scenario.scene.seed)
        detector = plume_detector()

        if self._link_real:
            tcp_port, udp_port = _free_port_pair()
            link_cfg: LinkConfig = replace(
                config.link, command_tcp_port=tcp_port, telemetry_udp_port=udp_port
            )
            config = replace(config, link=link_cfg)
            sim_inputs = SimDriverInputs(
                frames=frames,
                detector=detector,
                inbound_packets=[],
                thermal_readings=[20.0],
                power_readings=[10.0],
            )
            drivers = select_drivers(config, self._clock, sim_inputs)
            self._emulator = StationEmulator(
                tcp_host=link_cfg.command_tcp_host,
                tcp_port=link_cfg.command_tcp_port,
                udp_host=link_cfg.telemetry_udp_host,
                udp_port=link_cfg.telemetry_udp_port,
                key=_SIL_KEY,
                tc_apid=link_cfg.tc_apid,
            )
            self._emulator.connect()
        else:
            inbound = _scenario_packets(scenario, _SIL_KEY, config.link.tc_apid)
            sim_inputs = SimDriverInputs(
                frames=frames,
                detector=detector,
                inbound_packets=inbound,
                thermal_readings=[20.0],
                power_readings=[10.0],
            )
            drivers = select_drivers(config, self._clock, sim_inputs)

        calib = build_identity_calibration(config.sensor.height_px, config.sensor.width_px)
        apps = build_apps(
            config, self._bus, self._clock, drivers, MONITORED_SUBSYSTEMS, calib, _SIL_KEY
        )

        self._inf_sub = self._bus.subscribe(InferenceResultMsg)
        self._gimbal_sub = self._bus.subscribe(GimbalCommandMsg)
        self._mode_sub = self._bus.subscribe(ModeChangeMsg)
        self._ack_sub = self._bus.subscribe(CommandAckMsg)

        self._config = config
        self._apps = apps
        self._sensor = drivers.sensor
        self._gimbal = drivers.gimbal
        self._payload_state = apps.payload.controller.initial_state()
        self._fault_entries = apps.fault.initial_entries()
        self._apps_built = True

    def step(self, now: float) -> None:
        """Advance every subsystem one cycle via step_once, advancing the ManualClock first.

        Args:
            now: Monotonic seconds for the arbiter/watchdog (caller-advanced per step).

        Notes:
            The shared ManualClock is advanced to now so SimGimbal first-order dynamics
            integrate between steps (the closed loop only moves the gimbal across steps).
        """
        if not self._apps_built or self._payload_state is None:
            raise RuntimeError("build() must be called before step()")
        delta = now - self._clock.monotonic_s()
        if delta > 0.0:
            self._clock.advance(delta)
        self._payload_state, self._fault_entries = step_once(
            self._apps,  # type: ignore[arg-type]
            self._sensor,  # type: ignore[arg-type]
            self._gimbal,  # type: ignore[arg-type]
            self._bus,
            self._clock,
            now,
            self._payload_state,
            self._fault_entries,
        )

    def inject_command(self, step: CommandStep) -> None:
        """Send one command live for a real link; a no-op (pre-baked) for a sim link.

        Args:
            step: The command timeline entry to deliver.

        Notes:
            Sim-link commands are pre-baked into SimStationLink.inbound at build() time
            (its queue is fixed at construction), so this is intentionally a no-op there.
            Real-link commands are uplinked over TCP through the StationEmulator.
        """
        if self._link_real:
            if self._emulator is None:
                raise RuntimeError("real link backend has no StationEmulator")
            self._emulator.send_command(step.command_id, step.params, step.source, step.seq)

    def collect(self) -> TelemetryCapture:
        """Drain subscriptions + emulator UDP into a TelemetryCapture for scoring.

        Returns:
            TelemetryCapture: inference count, gimbal-moved flag, ordered mode changes,
            ordered ack statuses, and (real link only) the downlink datagrams the
            StationEmulator received.

        Notes:
            gimbal_moved compares the driver's authoritative read_position() against the
            origin with a tolerance (_GIMBAL_MOVED_TOLERANCE_DEG) that swamps SimGimbal's
            per-read encoder noise, so a stationary gimbal reliably reports False.
        """
        if (
            self._inf_sub is None
            or self._gimbal_sub is None
            or self._mode_sub is None
            or self._ack_sub is None
        ):
            raise RuntimeError("build() must be called before collect()")

        inference_count = 0
        while not self._inf_sub.empty():
            self._inf_sub.get_nowait()
            inference_count += 1

        while not self._gimbal_sub.empty():
            self._gimbal_sub.get_nowait()
        # Authoritative position from the gimbal driver: off-origin past the noise
        # tolerance == moved.
        gimbal_moved = False
        if self._gimbal is not None:
            read = self._gimbal.read_position()  # type: ignore[attr-defined]
            if not isinstance(read, Err):
                worst = max(abs(read.value.az_deg), abs(read.value.el_deg))
                gimbal_moved = worst > _GIMBAL_MOVED_TOLERANCE_DEG

        mode_changes = tuple(m.new_mode for m in self._drain(self._mode_sub))
        acks = tuple(a.status for a in self._drain(self._ack_sub))

        downlink: tuple[bytes, ...] = ()
        if self._emulator is not None:
            downlink = tuple(self._emulator.poll_downlink(timeout_s=0.2))

        return TelemetryCapture(
            inference_count=inference_count,
            gimbal_moved=gimbal_moved,
            mode_changes=mode_changes,
            acks=acks,
            downlink_packets=downlink,
        )

    def shutdown(self) -> None:
        """Close the StationEmulator (which closes the RealStationLink via app shutdown)."""
        if self._emulator is not None:
            self._emulator.close()
            self._emulator = None

    @staticmethod
    def _drain(subscription: Subscription[_T]) -> list[_T]:
        """Drain all pending messages from a subscription into a list (order-preserving)."""
        out: list[_T] = []
        while not subscription.empty():
            out.append(subscription.get_nowait())
        return out


class SocketBackend:
    """Deferred PIL/HIL transport backend: declared by the matrix, not implemented."""

    def build(self, scenario: Scenario, profile_path: str) -> None:
        """Raise: the PIL/HIL socket backend is defined-not-run for this milestone."""
        raise NotImplementedError("PIL/HIL socket backend deferred")

    def step(self, now: float) -> None:
        """Raise: the PIL/HIL socket backend is defined-not-run for this milestone."""
        raise NotImplementedError("PIL/HIL socket backend deferred")

    def inject_command(self, step: CommandStep) -> None:
        """Raise: the PIL/HIL socket backend is defined-not-run for this milestone."""
        raise NotImplementedError("PIL/HIL socket backend deferred")

    def collect(self) -> TelemetryCapture:
        """Raise: the PIL/HIL socket backend is defined-not-run for this milestone."""
        raise NotImplementedError("PIL/HIL socket backend deferred")

    def shutdown(self) -> None:
        """Raise: the PIL/HIL socket backend is defined-not-run for this milestone."""
        raise NotImplementedError("PIL/HIL socket backend deferred")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/gse/tests/test_harness_inprocess.py -v`  Expected: PASS (both tests green: the in-process backend collects 4 inferences and a moved gimbal beyond the 0.1 deg tolerance; SocketBackend.build raises NotImplementedError).

- [ ] **Step 5: Run gates**

```bash
uv run ruff check packages
uv run ruff format --check packages
uv run mypy packages
uv run lint-imports
uv run pytest packages/gse/tests -m "not e2e" -q
```
Expected: ruff clean (the `flight.libs.messages` import is pre-wrapped, so no E501; all lines <= 100 chars); `ruff format --check` clean (the pre-wrapped parenthesized import is already in ruff-format's canonical style, so no reformat is proposed); mypy clean (`_drain` uses the module-level `_T = TypeVar("_T")`, matching `bus.py`); `lint-imports` reports the `flight-gse-isolation` and `sim-gse-isolation` contracts kept (gse imports `flight.libs`, the `flight.core` composition seams, the `step_once`-threaded `flight.fault`/`flight.payload` types, and `sim` only — flight/sim do not import gse); gse tests pass.

- [ ] **Step 6: Commit**

```bash
git add packages/gse/src/gse/harness.py packages/gse/tests/test_harness_inprocess.py
git commit -m "feat(gse): in-process harness backend + deferred socket backend

InProcessBackend builds a PactConfig from the profile, renders the plume scene,
selects drivers via the composition-root seam, and steps the wired apps through
sim.sil.stepping.step_once. Real-link profiles stand up a RealStationLink +
StationEmulator over a free port pair; sim-link commands pre-bake into the
SimStationLink inbound queue. gimbal_moved uses a 0.1 deg tolerance over the
encoder-noise floor. SocketBackend is declared-not-run (NotImplementedError).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task C5: gse.orchestrator — run_scenario + ScenarioReport + assertion scoring

**Files:**
- Create: `packages/gse/src/gse/orchestrator.py`
- Create: `packages/gse/tests/test_orchestrator.py`

This task scores frame-portable assertions against the `TelemetryCapture` from a backend and records realtime-only assertions as `skip`-with-reason. Default backend is `InProcessBackend`. The frame-portable kinds per the contract are `mode_is`, `command_acked`, `gimbal_moved`, `min_inference_count`, `min_downlink_count`; the realtime-only kind is `ack_within_seconds`.

- [ ] **Step 1: Write the failing test**

```python
"""run_scenario scores frame-portable assertions and skips realtime-only ones."""

from gse.orchestrator import run_scenario
from gse.scenario import Assertion, Scenario, SceneSpec


def _scored_scenario() -> Scenario:
    """All-sim scenario asserting a moved gimbal, an inference floor, and a skipped timing one."""
    return Scenario(
        name="orchestrator-smoke",
        profile="profiles/sil.toml",
        scene=SceneSpec(num_frames=6, seed=0),
        commands=(),
        assertions=(
            Assertion(id="GIMBAL-MOVED", kind="gimbal_moved", value=True, tag="frame-portable"),
            Assertion(id="INF-FLOOR", kind="min_inference_count", value=6, tag="frame-portable"),
            Assertion(
                id="ACK-TIMING",
                kind="ack_within_seconds",
                value=2.0,
                tag="realtime-only",
            ),
        ),
        steps=6,
        dt=1.0,
    )


def test_run_scenario_scores_and_skips() -> None:
    """Frame-portable assertions pass; the realtime-only assertion is skipped with a reason."""
    report = run_scenario(_scored_scenario(), "profiles/sil.toml")

    assert report.scenario == "orchestrator-smoke"
    assert report.passed == 2
    assert report.failed == 0
    assert report.skipped == 1

    by_id = {r.id: r for r in report.results}
    assert by_id["GIMBAL-MOVED"].status == "pass"
    assert by_id["INF-FLOOR"].status == "pass"
    assert by_id["ACK-TIMING"].status == "skip"
    assert "realtime-only" in by_id["ACK-TIMING"].detail
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/gse/tests/test_orchestrator.py -v`  Expected: FAIL with `ModuleNotFoundError: No module named 'gse.orchestrator'`.

- [ ] **Step 3: Implement `gse/orchestrator.py`**

```python
"""GSE scenario orchestrator: run a scenario through a backend and score its assertions.

run_scenario builds the scenario on a HarnessBackend (default InProcessBackend), steps it
through its frame timeline, injects each command at its at_frame, collects a
TelemetryCapture, and scores every assertion. Frame-portable assertions are evaluated
against the deterministic capture; realtime-only assertions are recorded status="skip"
with a fixed reason (they are not meaningful under a ManualClock-driven in-process
backend -- they are reserved for the PIL/HIL socket backends).

Contains:
  - AssertionResult: per-assertion pass/fail/skip outcome with a detail string.
  - ScenarioReport: rolled-up counts + the ordered per-assertion results.
  - run_scenario: drive a scenario end-to-end and score it into a ScenarioReport.

Satisfies: REQ-COMM-HIGH-001, REQ-COMM-HIGH-003, REQ-GIMB-HIGH-001.
"""

from __future__ import annotations

# stdlib
from dataclasses import dataclass
from typing import Literal

# internal
from flight.libs.types import AckStatus, SystemMode
from gse.harness import HarnessBackend, InProcessBackend, TelemetryCapture
from gse.scenario import Assertion, Scenario

_SKIP_REASON = "realtime-only: not evaluated under deterministic in-process backend"
_FRAME_PORTABLE_KINDS = frozenset(
    {"mode_is", "command_acked", "gimbal_moved", "min_inference_count", "min_downlink_count"}
)


@dataclass(frozen=True, slots=True)
class AssertionResult:
    """Outcome of scoring one scenario assertion.

    Fields:
        id: The assertion's stable identifier (echoed from the scenario).
        tag: "frame-portable" or "realtime-only" (echoed from the scenario).
        status: "pass", "fail", or "skip".
        detail: Human-readable explanation (expected vs. observed, or the skip reason).
    """

    id: str
    tag: str
    status: Literal["pass", "fail", "skip"]
    detail: str


@dataclass(frozen=True, slots=True)
class ScenarioReport:
    """Rolled-up scoring report for one scenario run.

    Fields:
        scenario: The scenario name.
        passed: Count of frame-portable assertions that passed.
        failed: Count of frame-portable assertions that failed.
        skipped: Count of realtime-only assertions recorded as skipped.
        results: The ordered per-assertion results (one per scenario assertion).
    """

    scenario: str
    passed: int
    failed: int
    skipped: int
    results: tuple[AssertionResult, ...]


def _score_frame_portable(assertion: Assertion, capture: TelemetryCapture) -> AssertionResult:
    """Score one frame-portable assertion against a deterministic TelemetryCapture.

    Args:
        assertion: A frame-portable assertion (kind in _FRAME_PORTABLE_KINDS).
        capture: The collected telemetry to evaluate against.

    Returns:
        AssertionResult: pass/fail with an expected-vs-observed detail string.

    Notes:
        - mode_is: the scenario's terminal mode expectation. NOMINAL is satisfied iff NO
          SAFE was ever published; SAFE is satisfied iff at least one SAFE was published.
        - command_acked: an ACCEPTED/REJECTED ack of that status must appear in the run.
        - gimbal_moved: matches capture.gimbal_moved against the expected bool.
        - min_inference_count / min_downlink_count: observed >= the integer floor.
    """
    kind = assertion.kind
    if kind == "gimbal_moved":
        expected = bool(assertion.value)
        ok = capture.gimbal_moved == expected
        detail = f"expected gimbal_moved={expected}, got {capture.gimbal_moved}"
        return _result(assertion, ok, detail)
    if kind == "min_inference_count":
        floor = int(assertion.value)
        ok = capture.inference_count >= floor
        detail = f"inference_count {capture.inference_count} >= {floor}"
        return _result(assertion, ok, detail)
    if kind == "min_downlink_count":
        floor = int(assertion.value)
        observed = len(capture.downlink_packets)
        ok = observed >= floor
        detail = f"downlink_count {observed} >= {floor}"
        return _result(assertion, ok, detail)
    if kind == "command_acked":
        expected_status = AckStatus[str(assertion.value)]
        ok = expected_status in capture.acks
        detail = f"expected ack {expected_status.value} in {list(capture.acks)}"
        return _result(assertion, ok, detail)
    if kind == "mode_is":
        expected_mode = SystemMode[str(assertion.value)]
        saw_safe = SystemMode.SAFE in capture.mode_changes
        ok = saw_safe if expected_mode is SystemMode.SAFE else not saw_safe
        detail = f"expected mode {expected_mode.value}, safe_seen={saw_safe}"
        return _result(assertion, ok, detail)
    return _result(assertion, False, f"unknown frame-portable kind {kind!r}")


def _result(assertion: Assertion, ok: bool, detail: str) -> AssertionResult:
    """Build a pass/fail AssertionResult echoing the assertion id + tag."""
    return AssertionResult(
        id=assertion.id,
        tag=assertion.tag,
        status="pass" if ok else "fail",
        detail=detail,
    )


def run_scenario(
    scenario: Scenario,
    profile_path: str,
    backend: HarnessBackend | None = None,
) -> ScenarioReport:
    """Run a scenario through a backend and score every assertion into a ScenarioReport.

    Args:
        scenario: The scenario (scene spec, command timeline, assertions, steps, dt).
        profile_path: Path to the profile TOML override (selects the per-axis environment).
        backend: The HarnessBackend to run on; defaults to a fresh InProcessBackend.

    Returns:
        ScenarioReport: pass/fail/skip counts and the ordered per-assertion results.

    Notes:
        Steps the backend over scenario.steps cycles at scenario.dt seconds, injecting each
        command at its at_frame (1-based: injected just before the step that processes that
        frame). On a sim link inject_command is a no-op (commands are pre-baked and all
        ingest on step 1), so at_frame timing only takes effect on a real link. Frame-portable
        assertions are scored against the collected capture; realtime-only assertions are
        recorded status="skip" with a fixed reason. The backend is always shut down, even on a
        scoring error.
    """
    runner = backend if backend is not None else InProcessBackend()
    runner.build(scenario, profile_path)
    try:
        now = 0.0
        for frame in range(1, scenario.steps + 1):
            for step in scenario.commands:
                if step.at_frame == frame:
                    runner.inject_command(step)
            now += scenario.dt
            runner.step(now)
        capture = runner.collect()
    finally:
        runner.shutdown()

    results: list[AssertionResult] = []
    for assertion in scenario.assertions:
        if assertion.tag == "realtime-only" or assertion.kind not in _FRAME_PORTABLE_KINDS:
            results.append(
                AssertionResult(
                    id=assertion.id, tag=assertion.tag, status="skip", detail=_SKIP_REASON
                )
            )
            continue
        results.append(_score_frame_portable(assertion, capture))

    passed = sum(1 for r in results if r.status == "pass")
    failed = sum(1 for r in results if r.status == "fail")
    skipped = sum(1 for r in results if r.status == "skip")
    return ScenarioReport(
        scenario=scenario.name,
        passed=passed,
        failed=failed,
        skipped=skipped,
        results=tuple(results),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/gse/tests/test_orchestrator.py -v`  Expected: PASS (2 frame-portable assertions pass, the realtime-only one is skipped with the realtime-only reason).

- [ ] **Step 5: Run gates**

```bash
uv run ruff check packages
uv run ruff format --check packages
uv run mypy packages
uv run lint-imports
uv run pytest packages/gse/tests -m "not e2e" -q
```
Expected: ruff clean — every line in `orchestrator.py` is <= 100 chars (the `gimbal_moved` and `command_acked` detail f-strings are assembled on their own lines before `_result(...)`, so no E501); `ruff format --check`, mypy, and `lint-imports` clean; the gse-isolation contracts hold (orchestrator imports only `flight.libs` + `gse`).

- [ ] **Step 6: Commit**

```bash
git add packages/gse/src/gse/orchestrator.py packages/gse/tests/test_orchestrator.py
git commit -m "feat(gse): scenario orchestrator scores frame-portable assertions

run_scenario steps a HarnessBackend over a scenario's frame timeline, injects
commands at their at_frame, and scores mode_is/command_acked/gimbal_moved/
min_inference_count/min_downlink_count against the capture. Realtime-only
assertions (ack_within_seconds) are recorded skip-with-reason -- meaningless
under the deterministic in-process backend.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task C6: The blessed x86 partial — sil-link-real authenticated-command downlink test

**Files:**
- Create: `packages/gse/tests/test_sil_link_real.py`

This is the milestone integration: build the `sil-link-real` profile through `InProcessBackend` (so the link axis is `real` over a free TCP/UDP port pair, all other axes sim), uplink a signed `SET_THERMAL_LIMIT` through the `StationEmulator` with the SIL key, step until the command is routed and an `ACCEPTED` `CommandAckMsg` is downlinked as a CCSDS TM datagram, and assert the emulator's UDP socket received it. The loop is **event-counted and bounded** (a fixed max-step budget), never a wall-clock deadline — `ManualClock` makes wall-clock timing meaningless. The test is in default CI (NOT marked `e2e`).

The proof that the ACCEPTED ack reached the emulator is the downlink path: `IssIfaceApp.pump_downlink` serializes each `CommandAckMsg` to JSON (`{"type":"command_ack","status":"ACCEPTED",...}`), frames it as a CCSDS TM packet, and `RealStationLink.send_packet` sends it as UDP to the emulator's receiver. We decode the captured datagrams and assert one carries `status=ACCEPTED` for `SET_THERMAL_LIMIT`.

- [ ] **Step 1: Write the failing test**

```python
"""Blessed x86 partial: signed command uplinked over a real link, ACCEPTED ack downlinked."""

import json

from flight.libs.ccsds import decode_packet
from flight.libs.types import Ok
from gse.harness import InProcessBackend
from gse.scenario import CommandStep, Scenario, SceneSpec

_SIL_KEY = b"sil-test-key-0000000000000000000"


def _link_real_scenario() -> Scenario:
    """A sil-link-real scenario uplinking one signed SET_THERMAL_LIMIT at frame 1."""
    return Scenario(
        name="sil-link-real-ack",
        profile="profiles/sil-link-real.toml",
        scene=SceneSpec(num_frames=8, seed=0),
        commands=(
            CommandStep(
                at_frame=1,
                command_id="SET_THERMAL_LIMIT",
                params={"limit_c": 70.0},
                source="ground",
                seq=1,
            ),
        ),
        assertions=(),
        steps=8,
        dt=1.0,
    )


def _accepted_ack_downlinked(packets: list[bytes]) -> bool:
    """True if any captured TM datagram decodes to an ACCEPTED SET_THERMAL_LIMIT ack."""
    for raw in packets:
        decoded = decode_packet(raw)
        if not isinstance(decoded, Ok):
            continue
        _header, body = decoded.value
        try:
            record = json.loads(body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            continue
        if (
            record.get("type") == "command_ack"
            and record.get("status") == "ACCEPTED"
            and record.get("command_id") == "SET_THERMAL_LIMIT"
        ):
            return True
    return False


def test_sil_link_real_authenticated_command_acked_over_socket() -> None:
    """A signed command over a real socket link yields an ACCEPTED ack on the emulator's UDP."""
    backend = InProcessBackend()
    backend.build(_link_real_scenario(), "profiles/sil-link-real.toml")
    try:
        backend.inject_command(
            CommandStep(
                at_frame=1,
                command_id="SET_THERMAL_LIMIT",
                params={"limit_c": 70.0},
                source="ground",
                seq=1,
            )
        )
        # Event-counted, bounded poll loop (NOT a wall-clock deadline): step until the
        # emulator's UDP socket has captured the ACCEPTED downlink, up to a step budget.
        captured: list[bytes] = []
        found = False
        for i in range(20):
            backend.step(float(i + 1))
            captured.extend(backend.collect().downlink_packets)
            if _accepted_ack_downlinked(captured):
                found = True
                break
        assert found, f"no ACCEPTED SET_THERMAL_LIMIT ack in {len(captured)} captured datagrams"
    finally:
        backend.shutdown()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/gse/tests/test_sil_link_real.py -v`  Expected: FAIL initially with `ModuleNotFoundError`/`FileNotFoundError` if `profiles/sil-link-real.toml` or any upstream symbol (`gse.harness`, `select_drivers`, `step_once`, the relocated `build_tc_packet`) is not yet present — per the section-header ordering constraint, Phase A, Phase B, and C1-C5 MUST already be merged. Once they are, the test must instead reach the assertion. With all dependencies present, it fails ONLY if the real-link downlink path is broken (no ACCEPTED datagram captured). Confirm the dependency chain is in place before treating this as a real failure.

- [ ] **Step 3: No new implementation — this task is a pure integration assertion**

C6 introduces no new module: it exercises the `InProcessBackend` real-link path built in C4 (RealStationLink + StationEmulator over a free port pair) end-to-end. If the test fails after C4/C5 and the profile tasks are merged, debug the existing real-link wiring (e.g. `StationEmulator.connect()` must complete so `RealStationLink.link_state()` is AOS before `pump_downlink` runs; `_free_port_pair` must yield a port the emulator can connect to). No code is added here unless a defect in C4's backend is found, in which case the fix lands in `packages/gse/src/gse/harness.py` (the regression guard is this test plus `test_harness_inprocess.py`).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/gse/tests/test_sil_link_real.py -v`  Expected: PASS — within the 20-step budget the emulator's UDP receiver captures a CCSDS TM datagram decoding to `{"type":"command_ack","status":"ACCEPTED","command_id":"SET_THERMAL_LIMIT",...}`.

- [ ] **Step 5: Run gates**

```bash
uv run ruff check packages
uv run ruff format --check packages
uv run mypy packages
uv run lint-imports
uv run pytest packages/gse/tests -m "not e2e" -q
```
Expected: all clean; the full gse suite (harness, orchestrator, sil-link-real) passes under `-m "not e2e"`, confirming the blessed partial runs in default CI.

- [ ] **Step 6: Commit**

```bash
git add packages/gse/tests/test_sil_link_real.py
git commit -m "test(gse): blessed x86 partial -- signed command acked over a real socket link

Builds the sil-link-real profile through InProcessBackend (real link over a free
TCP/UDP port pair, other axes sim), uplinks a signed SET_THERMAL_LIMIT via the
StationEmulator, and steps a bounded event-counted loop until the ACCEPTED
CommandAckMsg is downlinked as a CCSDS TM datagram and captured on the emulator's
UDP socket. Not marked e2e -- runs in default CI.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

## Phase D: import contracts + VCRM + traceability CI check + PIL/HIL docs

> **Orchestration / phase-ordering constraints for Phase D (read first).** These are hard gates,
> not advisory. The orchestrator MUST enforce them; do not start a Phase-D task whose precondition
> is unmet.
>
> - **Phase A MUST land before Phase C, and Phase C before Phase D.** The `pact-gse` package (C2
>   `StationEmulator`, C3, C4 harness) imports `build_tc_packet` from `flight.libs.commands`, which
>   only exists after Phase A relocates it there (the as-built `flight.libs.commands.__init__`
>   exports only the dictionary symbols). Running C before A makes every gse import raise
>   `ImportError`; running D's gate (D5) before C leaves the gse tests uncollectable.
> - **C1 MUST run before D1.** `uv run lint-imports` in D1 can only resolve the new `gse` root
>   package and the two forbidden contracts once `pact-gse` is editable-installed via the uv
>   workspace. C1 owns registering `pact-gse` in `[tool.uv.workspace]` members, `[tool.uv.sources]`,
>   the dev extras, and running `uv sync --extra dev`. If D1 runs first, `lint-imports` fails with
>   "could not find package gse".
> - **REQ-ID verification is a required step in D2, not optional.** Every ID seeded in a
>   requirement's `modules` list was confirmed (via `grep -rn "Satisfies:" packages/flight/src`)
>   to be cited verbatim by a flight module: REQ-COMM-HIGH-001 (`iss_iface/app.py`,
>   `drivers_real/station.py`, `enums.py`), REQ-COMM-HIGH-002 (`ccsds/codec.py`,
>   `drivers_real/station.py`), REQ-COMM-HIGH-003 (`iss_iface/app.py`, `commands/dictionary.py`,
>   `ingress/pipeline.py`, `enums.py`), REQ-COMM-HIGH-004 (`iss_iface/app.py`,
>   `ingress/pipeline.py`, `enums.py`), REQ-SAFE-HIGH-002 (`thermal/app.py`, `electrical/app.py`,
>   `fault/watchdog.py`, `fault/app.py`, `fault/policy.py`), REQ-AIML-GIMB-001 (`config.py`,
>   `gimbal/arbiter.py`, both gimbal drivers), REQ-GIMB-HIGH-001 (`payload/control.py`,
>   `config.py`, `gimbal/arbiter.py`), REQ-GIMB-HIGH-003 (`gimbal/runaway.py`, `fault/policy.py`).
>   If a future edit changes the seed, re-run that grep and drop/remap any uncited ID before
>   landing D2 -- do not fabricate IDs.
> - **C4 convention note (not a Phase-D task):** C4's `_drain` helper must use the explicit
>   `Generic`/module-level `TypeVar` form to match `flight.libs.bus.Subscription` (which keeps the
>   older `Generic[_T]` style with a `# noqa: UP046` justification), not PEP 695
>   `def _drain[T](...)` syntax. This is enforced in C4; it is restated here only so the Phase-D
>   gate (D5) treats a PEP 695 occurrence as a C-phase regression to fix in C4, not in D.

### Task D1: Add `flight-gse-isolation` and `sim-gse-isolation` import contracts

**Files:**
- Modify: `.importlinter` (root_packages block lines 1-5; append two new contract blocks after line 70)
- Test: `packages/sim/tests/test_import_contracts_gse.py` (Create)

> **Precondition:** C1 must have landed first (see orchestration constraints above) so `pact-gse`
> is editable-installed and `lint-imports` can resolve the `gse` root package.
>
> Pure-config + lint task. The regression guard is the live `uv run lint-imports` run plus a small
> static parser test that asserts the two new contract stanzas exist with the correct
> source/forbidden modules. The `gse` package itself and its addition to `[tool.uv.workspace]`
> members come from earlier phases; this task only adds the import-linter `root_packages` entry and
> the two forbidden contracts, and proves they are enforced.
>
> **CI-collection note:** this guard test lives under `packages/sim/tests/` (not the repo-root
> `tests/`) precisely so the existing CI `Tests` step (`uv run pytest packages -m "not e2e"`)
> collects it. CI scopes pytest to `packages/`, so a root-`tests/` placement would never run in CI.

- [ ] **Step 1: Write the failing test**
```python
# packages/sim/tests/test_import_contracts_gse.py
"""Static checks that the GSE isolation import contracts are declared.

Guards that flight and sim cannot import the gse package, enforced by import-linter.
"""

from __future__ import annotations

import configparser
from pathlib import Path


def _importlinter_path() -> Path:
    """Locate the repo-root .importlinter by walking up until it is found."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / ".importlinter"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("could not locate .importlinter above the test file")


def _load() -> configparser.ConfigParser:
    """Parse the repo-root .importlinter INI file."""
    parser = configparser.ConfigParser()
    parser.read(_importlinter_path(), encoding="utf-8")
    return parser


def test_gse_is_a_root_package() -> None:
    """gse must be registered as a root package so import-linter scans it."""
    parser = _load()
    roots = parser["importlinter"]["root_packages"].split()
    assert "gse" in roots


def test_flight_gse_isolation_contract_declared() -> None:
    """flight must be forbidden from importing gse."""
    section = "importlinter:contract:flight-gse-isolation"
    parser = _load()
    assert parser.has_section(section)
    assert parser[section]["type"].strip() == "forbidden"
    assert parser[section]["source_modules"].split() == ["flight"]
    assert parser[section]["forbidden_modules"].split() == ["gse"]


def test_sim_gse_isolation_contract_declared() -> None:
    """sim must be forbidden from importing gse."""
    section = "importlinter:contract:sim-gse-isolation"
    parser = _load()
    assert parser.has_section(section)
    assert parser[section]["type"].strip() == "forbidden"
    assert parser[section]["source_modules"].split() == ["sim"]
    assert parser[section]["forbidden_modules"].split() == ["gse"]
```

- [ ] **Step 2: Run test to verify it fails**
Run: `uv run pytest packages/sim/tests/test_import_contracts_gse.py -v`
Expected: FAIL — `test_gse_is_a_root_package` (gse not yet in root_packages) and both contract tests fail with `assert parser.has_section(section)` being False (sections not yet declared).

- [ ] **Step 3: Add `gse` to root_packages and append the two contracts**

Edit the `root_packages` block in `.importlinter` from:
```ini
[importlinter]
root_packages =
    flight
    sim
    tools
```
to:
```ini
[importlinter]
root_packages =
    flight
    sim
    tools
    gse
```

Then append the two new contract blocks to the end of `.importlinter` (after the existing
`drivers-independent-reverse` contract):
```ini

[importlinter:contract:flight-gse-isolation]
name = Flight must not import the ground-support emulator
type = forbidden
source_modules =
    flight
forbidden_modules =
    gse

[importlinter:contract:sim-gse-isolation]
name = Sim must not import the ground-support emulator
type = forbidden
source_modules =
    sim
forbidden_modules =
    gse
```

- [ ] **Step 4: Run test to verify it passes**
Run: `uv run pytest packages/sim/tests/test_import_contracts_gse.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Run gates**
```bash
uv run lint-imports
uv run ruff check packages
uv run pytest packages/sim/tests/test_import_contracts_gse.py -v
```
Expected: `lint-imports` reports all contracts kept (including the two new ones); ruff clean; tests pass.
Note: `lint-imports` requires the `gse` package to be importable (installed editable via the workspace), which C1 provides; if it reports "could not find package gse", C1 has not run -- fix the phase ordering, not this task.

- [ ] **Step 6: Commit**
```bash
git add .importlinter packages/sim/tests/test_import_contracts_gse.py
git commit -m "build(importlinter): forbid flight/sim from importing gse

Add gse as a scanned root package and declare flight-gse-isolation +
sim-gse-isolation forbidden contracts so the ground-support emulator
stays a one-way leaf (it may import flight.libs + sim, never the reverse).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task D2: Seed the VCRM (`vcrm.toml` + rendered `vcrm.md`)

**Files:**
- Create: `docs/requirements/vcrm.toml`
- Create: `docs/requirements/vcrm.md`
- Test: `packages/sim/tests/test_vcrm_seed_integrity.py` (Create)

> The VCRM is the machine-readable traceability source of truth. We seed ONLY the thin slice of
> requirements actually exercised by the two RUNNING profiles (`sil` and `sil-link-real`):
> command-ingress auth, ACK/NACK, CCSDS framing, AOS/LOS gating, SAFE-on-thermal, and closed-loop
> pointing. Every `modules` REQ-ID below was confirmed (grep) to be cited by a real
> `# Satisfies: REQ-...` header (see the orchestration constraints at the top of this section for
> the per-ID confirmation). **Required step before editing the seed:** re-run
> `grep -rn "Satisfies:" packages/flight/src` and confirm each seeded `modules` ID appears verbatim;
> do NOT add any REQ-ID not already cited by a module. The permanent gap row (real ground segment
> never tested; GSE stands in) is recorded with `status = "gap"` and `venue = "none"`.
>
> **CI-collection note:** the guard test lives under `packages/sim/tests/` so the CI `Tests` step
> (`uv run pytest packages -m "not e2e"`) collects it; a repo-root `tests/` placement would not run
> in CI. The test resolves the repo root by walking up to the directory that contains
> `docs/requirements/vcrm.toml`, so it is location-independent.

- [ ] **Step 1: Write the failing test**
```python
# packages/sim/tests/test_vcrm_seed_integrity.py
"""Integrity checks on the seeded VCRM source of truth.

Asserts vcrm.toml parses, only seeds RUNNING-venue requirements (plus the permanent gap),
and that every cited Satisfies: REQ-ID actually appears in a flight module docstring.
"""

from __future__ import annotations

import tomllib
from pathlib import Path


def _repo_root() -> Path:
    """Walk up from this test file to the directory holding docs/requirements/vcrm.toml."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "docs" / "requirements" / "vcrm.toml").exists():
            return parent
    raise FileNotFoundError("could not locate docs/requirements/vcrm.toml above the test file")


_RUNNING_VENUES = {"unit", "sil", "sil-link-real"}
_ALL_VENUES = _RUNNING_VENUES | {"pil", "hil", "none"}
_METHODS = {"unit", "SIL", "PIL", "HIL", "none"}


def _vcrm_path() -> Path:
    """Absolute path to the seeded vcrm.toml."""
    return _repo_root() / "docs" / "requirements" / "vcrm.toml"


def _flight_src() -> Path:
    """Absolute path to the flight source tree."""
    return _repo_root() / "packages" / "flight" / "src"


def _load() -> dict:
    """Parse vcrm.toml into a dict."""
    with _vcrm_path().open("rb") as handle:
        return tomllib.load(handle)


def _cited_req_ids() -> set[str]:
    """Collect every REQ-ID appearing after a 'Satisfies:' marker in flight sources."""
    found: set[str] = set()
    for path in _flight_src().rglob("*.py"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if "Satisfies:" not in line:
                continue
            tail = line.split("Satisfies:", 1)[1]
            for token in tail.replace(",", " ").split():
                if token.startswith("REQ-"):
                    found.add(token.strip(".() "))
    return found


def test_vcrm_parses_and_has_requirements() -> None:
    """vcrm.toml must parse and contain at least one requirement."""
    data = _load()
    assert isinstance(data.get("requirement"), list)
    assert len(data["requirement"]) >= 6


def test_every_requirement_has_required_fields() -> None:
    """Each requirement entry carries the full schema with valid enum values."""
    for req in _load()["requirement"]:
        for key in ("id", "statement", "method", "venue", "modules", "evidence", "status"):
            assert key in req, f"{req.get('id')} missing {key}"
        assert req["method"] in _METHODS
        assert req["venue"] in _ALL_VENUES
        assert isinstance(req["modules"], list)
        assert isinstance(req["evidence"], list)


def test_only_running_venues_or_permanent_gap() -> None:
    """Seeded requirements target a running venue, or are the recorded permanent gap."""
    for req in _load()["requirement"]:
        if req["status"] == "gap":
            assert req["venue"] == "none"
        else:
            assert req["venue"] in _RUNNING_VENUES


def test_cited_modules_actually_exist_in_source() -> None:
    """Every REQ-ID listed in a requirement's modules is cited by a flight module."""
    cited = _cited_req_ids()
    for req in _load()["requirement"]:
        for req_id in req["modules"]:
            assert req_id in cited, f"{req_id} not cited by any flight Satisfies: header"


def test_permanent_ground_segment_gap_present() -> None:
    """The permanent 'real ground segment never tested' gap row must exist."""
    gaps = [r for r in _load()["requirement"] if r["status"] == "gap"]
    assert any("ground segment" in r["statement"].lower() for r in gaps)
```

- [ ] **Step 2: Run test to verify it fails**
Run: `uv run pytest packages/sim/tests/test_vcrm_seed_integrity.py -v`
Expected: FAIL — `docs/requirements/vcrm.toml` does not exist, so `_repo_root()` raises `FileNotFoundError` during collection/first call.

- [ ] **Step 3: Create `vcrm.toml` and `vcrm.md`**

Create `docs/requirements/vcrm.toml`:
```toml
# PACT Verification Cross-Reference Matrix (VCRM)
#
# Machine-readable source of truth for requirement -> verification-venue traceability.
# Source of truth: this file. The rendered table (vcrm.md) is derived from it.
#
# Thin slice: this VCRM seeds ONLY requirements actually exercised by the two RUNNING
# validation profiles -- sil (profiles/sil.toml) and sil-link-real (profiles/sil-link-real.toml).
# Requirements verified only at PIL/HIL are intentionally NOT marked "verified" here because
# those venues are DEFINED-NOT-RUN (no hardware yet). scripts/check_vcrm.py enforces both rules.
#
# Field schema per [[requirement]]:
#   id        : requirement identifier (mirrors the "Satisfies:" REQ-ID cited in module docstrings)
#   statement : one-line plain-English requirement statement
#   method    : verification method -- "unit" | "SIL" | "PIL" | "HIL" | "none"
#   venue     : where evidence is produced -- "unit" | "sil" | "sil-link-real" | "pil" | "hil" | "none"
#   modules   : list of REQ-IDs to grep for as "Satisfies:" markers in packages/flight/src
#   evidence  : list of test ids / scenario ids that exercise the requirement
#   status    : "verified" | "partial" | "gap"

[[requirement]]
id = "REQ-COMM-HIGH-003"
statement = "Authenticated command ingress: only HMAC-valid commands from accepted sources are accepted."
method = "SIL"
venue = "sil"
modules = ["REQ-COMM-HIGH-003"]
evidence = ["packages/flight/tests/test_iss_ingress_pipeline.py", "scenario:ingress_auth_accept"]
status = "verified"

[[requirement]]
id = "REQ-COMM-HIGH-004"
statement = "Command acknowledgement: every ingested command yields an ACCEPTED or REJECTED ack."
method = "SIL"
venue = "sil"
modules = ["REQ-COMM-HIGH-004"]
evidence = ["packages/flight/tests/test_iss_iface_app.py", "scenario:ingress_nack_bad_hmac"]
status = "verified"

[[requirement]]
id = "REQ-COMM-HIGH-002"
statement = "CCSDS framing with CRC integrity on uplink and downlink packets."
method = "SIL"
venue = "sil"
modules = ["REQ-COMM-HIGH-002"]
evidence = ["packages/flight/tests/test_ccsds_codec.py", "scenario:downlink_ccsds_frames"]
status = "verified"

[[requirement]]
id = "REQ-COMM-HIGH-001"
statement = "Downlink is gated by link visibility: telemetry is emitted only during AOS."
method = "SIL"
venue = "sil-link-real"
modules = ["REQ-COMM-HIGH-001"]
evidence = ["packages/sim/tests/test_sil_closed_loop.py", "scenario:aos_los_gating"]
status = "verified"

[[requirement]]
id = "REQ-SAFE-HIGH-002"
statement = "Thermal over-limit drives the system to SAFE mode and stows the gimbal."
method = "SIL"
venue = "sil"
modules = ["REQ-SAFE-HIGH-002"]
evidence = ["packages/sim/tests/test_sil_closed_loop.py", "scenario:safe_on_thermal"]
status = "verified"

[[requirement]]
id = "REQ-AIML-GIMB-001"
statement = "Autonomous closed-loop pointing rates the gimbal toward the detected plume."
method = "SIL"
venue = "sil"
modules = ["REQ-AIML-GIMB-001"]
evidence = ["packages/sim/tests/test_sil_closed_loop.py", "scenario:closed_loop_pointing"]
status = "verified"

[[requirement]]
id = "REQ-GIMB-HIGH-001"
statement = "Region-of-interest retention keeps the plume within the pointing deadband."
method = "SIL"
venue = "sil"
modules = ["REQ-GIMB-HIGH-001"]
evidence = ["packages/sim/tests/test_sil_closed_loop.py", "scenario:closed_loop_pointing"]
status = "verified"

[[requirement]]
id = "REQ-GIMB-HIGH-003"
statement = "Runaway / unsafe gimbal motion is detected and forces a stow."
method = "SIL"
venue = "sil"
modules = ["REQ-GIMB-HIGH-003"]
evidence = ["packages/flight/tests/test_runaway.py", "scenario:safe_on_thermal"]
status = "verified"

# --- PERMANENT GAP (recorded, never closed by the in-process/SIL venues) -----------------
[[requirement]]
id = "GAP-GROUND-SEGMENT"
statement = "Real ground segment is never tested; the GSE station emulator stands in for it."
method = "none"
venue = "none"
modules = []
evidence = []
status = "gap"
```

Create `docs/requirements/vcrm.md`:
```markdown
# PACT Verification Cross-Reference Matrix (VCRM)

> **Source of truth:** `docs/requirements/vcrm.toml`. This table is rendered from it.
> CI (`scripts/check_vcrm.py`) enforces that every running-venue requirement is both cited by a
> module docstring (`Satisfies:`) and backed by evidence, and that no PIL/HIL requirement claims
> `verified`.

## Scope (thin slice)

This matrix covers only requirements exercised by the two **running** validation profiles:

| Profile | File | Venue |
| --- | --- | --- |
| SIL (full sim) | `profiles/sil.toml` | `sil` |
| SIL + real link | `profiles/sil-link-real.toml` | `sil-link-real` |

PIL and HIL profiles are **DEFINED-NOT-RUN** (no hardware yet); requirements verifiable only there
are deliberately absent rather than falsely marked verified.

## Matrix

| Requirement | Statement | Method | Venue | Evidence | Status |
| --- | --- | --- | --- | --- | --- |
| REQ-COMM-HIGH-003 | Authenticated command ingress (HMAC + accepted sources) | SIL | sil | test_iss_ingress_pipeline; scenario:ingress_auth_accept | verified |
| REQ-COMM-HIGH-004 | Command ACK/NACK for every ingested command | SIL | sil | test_iss_iface_app; scenario:ingress_nack_bad_hmac | verified |
| REQ-COMM-HIGH-002 | CCSDS framing + CRC integrity | SIL | sil | test_ccsds_codec; scenario:downlink_ccsds_frames | verified |
| REQ-COMM-HIGH-001 | Downlink gated by AOS visibility | SIL | sil-link-real | test_sil_closed_loop; scenario:aos_los_gating | verified |
| REQ-SAFE-HIGH-002 | Thermal over-limit -> SAFE + stow | SIL | sil | test_sil_closed_loop; scenario:safe_on_thermal | verified |
| REQ-AIML-GIMB-001 | Autonomous closed-loop pointing toward plume | SIL | sil | test_sil_closed_loop; scenario:closed_loop_pointing | verified |
| REQ-GIMB-HIGH-001 | ROI retention within pointing deadband | SIL | sil | test_sil_closed_loop; scenario:closed_loop_pointing | verified |
| REQ-GIMB-HIGH-003 | Runaway gimbal detection forces stow | SIL | sil | test_runaway; scenario:safe_on_thermal | verified |

## Permanent gaps

| Gap | Statement | Status |
| --- | --- | --- |
| GAP-GROUND-SEGMENT | Real ground segment is never tested; the GSE station emulator stands in for it. The `lock` axis (LaunchLock) is likewise a permanent VCRM gap -- there is no device and no config field, only this record. | gap |
```

- [ ] **Step 4: Run test to verify it passes**
Run: `uv run pytest packages/sim/tests/test_vcrm_seed_integrity.py -v`
Expected: PASS (6 passed). If `test_cited_modules_actually_exist_in_source` fails, re-run
`grep -rn "Satisfies:" packages/flight/src` and align `modules` to actually-cited IDs — do not
fabricate.

- [ ] **Step 5: Run gates**
```bash
uv run pytest packages/sim/tests/test_vcrm_seed_integrity.py -v
```
Expected: PASS. (No ruff/mypy needed for the `.toml`/`.md` data; the test file is linted/typed by the whole-suite run in D5.)

- [ ] **Step 6: Commit**
```bash
git add docs/requirements/vcrm.toml docs/requirements/vcrm.md packages/sim/tests/test_vcrm_seed_integrity.py
git commit -m "docs(vcrm): seed traceability matrix for running SIL profiles

Machine-readable vcrm.toml (source of truth) + rendered vcrm.md covering the
thin slice of requirements exercised by sil + sil-link-real (command auth,
ACK/NACK, CCSDS framing, AOS/LOS gating, SAFE-on-thermal, closed-loop
pointing). Records the permanent 'real ground segment never tested' gap.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task D3: VCRM traceability CI check (`scripts/check_vcrm.py`) + CI steps

**Files:**
- Create: `scripts/check_vcrm.py`
- Create: `packages/sim/tests/test_check_vcrm.py`
- Modify: `.github/workflows/ci.yml` (add a `VCRM traceability` step after "Import layering", line 26; add a `Root guard tests` step after the existing "Tests" step, line 30)

> `check_vcrm.py` is stdlib-only (`tomllib`, `argparse`, `pathlib`, `sys`) so it runs without the
> workspace installed. It parses `vcrm.toml` and asserts: (1) every requirement whose venue is a
> RUNNING profile (`unit`/`sil`/`sil-link-real`) is cited by >=1 module docstring
> (`Satisfies: <ID>`) AND has >=1 evidence entry; (2) NO requirement with venue `pil`/`hil` claims
> `status = "verified"`. Exit nonzero (with a printed reason) on any violation. The test drives it
> on the seeded file (exit 0) and a malformed fixture (nonzero).
>
> **This task also closes the CI-collection gap (reviewer finding 5).** The four Phase-D guard
> tests now live under `packages/sim/tests/` so the existing `uv run pytest packages -m "not e2e"`
> step collects them. As a belt-and-suspenders measure (and to make the intent explicit for any
> future root-`tests/` additions), this task also adds a dedicated CI step that runs the repo-root
> `tests/` directory with the `not e2e` marker, so root-level guard tests can never silently escape
> CI again.

- [ ] **Step 1: Write the failing test**
```python
# packages/sim/tests/test_check_vcrm.py
"""Drive scripts/check_vcrm.py against the seeded VCRM and malformed fixtures."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    """Walk up to the directory holding scripts/check_vcrm.py."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "scripts" / "check_vcrm.py").exists():
            return parent
    raise FileNotFoundError("could not locate scripts/check_vcrm.py above the test file")


def _script() -> Path:
    """Absolute path to the check_vcrm.py script under test."""
    return _repo_root() / "scripts" / "check_vcrm.py"


def _seeded() -> Path:
    """Absolute path to the real seeded vcrm.toml."""
    return _repo_root() / "docs" / "requirements" / "vcrm.toml"


def _run(vcrm_path: Path, src_root: Path) -> subprocess.CompletedProcess[str]:
    """Invoke check_vcrm.py with explicit --vcrm and --src arguments."""
    return subprocess.run(
        [sys.executable, str(_script()), "--vcrm", str(vcrm_path), "--src", str(src_root)],
        capture_output=True,
        text=True,
    )


def test_seeded_vcrm_passes() -> None:
    """The real seeded vcrm.toml against real flight sources exits 0."""
    result = _run(_seeded(), _repo_root() / "packages" / "flight" / "src")
    assert result.returncode == 0, result.stdout + result.stderr


def test_running_requirement_without_citation_fails(tmp_path: Path) -> None:
    """A running-venue requirement citing an uncited REQ-ID exits nonzero."""
    fake_src = tmp_path / "src"
    fake_src.mkdir()
    (fake_src / "mod.py").write_text('"""Satisfies: REQ-REAL-001."""\n', encoding="utf-8")
    vcrm = tmp_path / "vcrm.toml"
    vcrm.write_text(
        '[[requirement]]\n'
        'id = "REQ-FAKE-999"\n'
        'statement = "uncited"\n'
        'method = "SIL"\n'
        'venue = "sil"\n'
        'modules = ["REQ-FAKE-999"]\n'
        'evidence = ["t"]\n'
        'status = "verified"\n',
        encoding="utf-8",
    )
    result = _run(vcrm, fake_src)
    assert result.returncode != 0
    assert "REQ-FAKE-999" in result.stdout


def test_running_requirement_without_evidence_fails(tmp_path: Path) -> None:
    """A running-venue requirement with empty evidence exits nonzero."""
    fake_src = tmp_path / "src"
    fake_src.mkdir()
    (fake_src / "mod.py").write_text('"""Satisfies: REQ-REAL-001."""\n', encoding="utf-8")
    vcrm = tmp_path / "vcrm.toml"
    vcrm.write_text(
        '[[requirement]]\n'
        'id = "REQ-REAL-001"\n'
        'statement = "cited but no evidence"\n'
        'method = "SIL"\n'
        'venue = "sil"\n'
        'modules = ["REQ-REAL-001"]\n'
        'evidence = []\n'
        'status = "verified"\n',
        encoding="utf-8",
    )
    result = _run(vcrm, fake_src)
    assert result.returncode != 0
    assert "evidence" in result.stdout.lower()


def test_pil_hil_verified_claim_fails(tmp_path: Path) -> None:
    """A pil/hil requirement claiming verified exits nonzero (non-running venue)."""
    fake_src = tmp_path / "src"
    fake_src.mkdir()
    (fake_src / "mod.py").write_text('"""Satisfies: REQ-HW-001."""\n', encoding="utf-8")
    vcrm = tmp_path / "vcrm.toml"
    vcrm.write_text(
        '[[requirement]]\n'
        'id = "REQ-HW-001"\n'
        'statement = "hardware only"\n'
        'method = "HIL"\n'
        'venue = "hil"\n'
        'modules = ["REQ-HW-001"]\n'
        'evidence = ["t"]\n'
        'status = "verified"\n',
        encoding="utf-8",
    )
    result = _run(vcrm, fake_src)
    assert result.returncode != 0
    assert "hil" in result.stdout.lower()
```

- [ ] **Step 2: Run test to verify it fails**
Run: `uv run pytest packages/sim/tests/test_check_vcrm.py -v`
Expected: FAIL — `scripts/check_vcrm.py` does not exist, so `_repo_root()` raises `FileNotFoundError` (the script marker is not found) during the first test call.

- [ ] **Step 3: Create `scripts/check_vcrm.py`**
```python
#!/usr/bin/env python3
"""VCRM traceability CI check.

Satisfies: REQ-OPER-HIGH-002 (verifiable, type-safe operational config and traceability).

Parses docs/requirements/vcrm.toml (stdlib tomllib) and enforces two invariants:
  1. Every requirement whose venue is a RUNNING profile (unit | sil | sil-link-real) is cited by
     at least one module docstring ("Satisfies: <ID>") under the flight source tree AND has at
     least one evidence entry.
  2. No requirement whose venue is a non-running profile (pil | hil) claims status="verified".

Exits 0 when both invariants hold; otherwise prints each violation and exits 1. Stdlib only so it
runs in CI without the uv workspace installed.
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

_RUNNING_VENUES = frozenset({"unit", "sil", "sil-link-real"})
_NON_RUNNING_VENUES = frozenset({"pil", "hil"})


def _collect_cited_ids(src_root: Path) -> set[str]:
    """Return every REQ-ID following a 'Satisfies:' marker under src_root.

    Args:
        src_root: directory tree to scan for *.py files.

    Returns:
        Set of REQ-ID strings (tokens beginning with 'REQ-').
    """
    cited: set[str] = set()
    for path in src_root.rglob("*.py"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if "Satisfies:" not in line:
                continue
            tail = line.split("Satisfies:", 1)[1]
            for token in tail.replace(",", " ").split():
                stripped = token.strip(".() ")
                if stripped.startswith("REQ-"):
                    cited.add(stripped)
    return cited


def _check(vcrm_path: Path, src_root: Path) -> list[str]:
    """Validate the VCRM and return a list of human-readable violation strings.

    Args:
        vcrm_path: path to vcrm.toml.
        src_root: flight source tree to scan for Satisfies: citations.

    Returns:
        List of violation messages; empty list means the VCRM is consistent.
    """
    with vcrm_path.open("rb") as handle:
        data = tomllib.load(handle)
    cited = _collect_cited_ids(src_root)
    violations: list[str] = []
    for req in data.get("requirement", []):
        req_id = req.get("id", "<missing-id>")
        venue = req.get("venue", "none")
        status = req.get("status", "gap")
        if venue in _RUNNING_VENUES:
            modules = req.get("modules", [])
            if not modules:
                violations.append(f"{req_id}: running venue '{venue}' but no modules listed")
            for module_id in modules:
                if module_id not in cited:
                    violations.append(
                        f"{req_id}: cites module {module_id} not found in any "
                        f"'Satisfies:' header under {src_root}"
                    )
            if not req.get("evidence", []):
                violations.append(f"{req_id}: running venue '{venue}' but no evidence listed")
        if venue in _NON_RUNNING_VENUES and status == "verified":
            violations.append(
                f"{req_id}: venue '{venue}' is not a running profile but status='verified'"
            )
    return violations


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the VCRM check, and return a process exit code.

    Args:
        argv: optional argument vector (defaults to sys.argv[1:]).

    Returns:
        0 if the VCRM is consistent, 1 otherwise.
    """
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Check VCRM traceability invariants.")
    parser.add_argument(
        "--vcrm",
        type=Path,
        default=repo_root / "docs" / "requirements" / "vcrm.toml",
        help="Path to vcrm.toml.",
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=repo_root / "packages" / "flight" / "src",
        help="Flight source tree to scan for Satisfies: citations.",
    )
    args = parser.parse_args(argv)
    violations = _check(args.vcrm, args.src)
    if violations:
        print("VCRM check FAILED:")
        for line in violations:
            print(f"  - {line}")
        return 1
    print("VCRM check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**
Run: `uv run pytest packages/sim/tests/test_check_vcrm.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Add the CI steps and run gates**

Edit `.github/workflows/ci.yml`. After the "Import layering" step (lines 25-26):
```yaml
      - name: Import layering
        run: uv run lint-imports
```
insert:
```yaml
      - name: VCRM traceability
        run: uv run python scripts/check_vcrm.py
```

Then, after the existing "Tests" step (the `uv run pytest packages -m "not e2e"` step, lines 27-30),
append a step that runs the repo-root `tests/` guard suite so any root-level guard test is also
collected in CI (closes the gap where `pytest packages` never scanned `tests/`):
```yaml
      - name: Root guard tests
        # Repo-root tests/ holds cross-cutting guards (e.g. legacy regression guards) that the
        # packages-scoped "Tests" step does not collect. Run them explicitly with the same marker.
        run: uv run pytest tests -m "not e2e"
```

Run gates:
```bash
uv run python scripts/check_vcrm.py
uv run ruff check packages scripts
uv run ruff format --check packages scripts
uv run pytest packages/sim/tests/test_check_vcrm.py -v
```
Expected: `check_vcrm.py` prints "VCRM check passed." and exits 0; ruff/format clean; tests pass.
Note: `scripts/` is outside `packages/`; include it explicitly in these local ruff invocations so the new module is linted. The stdlib-only script is type-checked indirectly via its passing tests; mypy of `scripts/` is not in the workspace mypy path and is not required here.

- [ ] **Step 6: Commit**
```bash
git add scripts/check_vcrm.py packages/sim/tests/test_check_vcrm.py .github/workflows/ci.yml
git commit -m "ci(vcrm): enforce requirement traceability and collect root guard tests in CI

scripts/check_vcrm.py (stdlib + tomllib) asserts every running-venue
requirement is cited by a module Satisfies: header and backed by evidence,
and that no pil/hil requirement claims verified. Wire it as a CI gate and add
a dedicated 'Root guard tests' step so repo-root tests/ guards run in CI too.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task D4: PIL/HIL procedure docs + CONTEXT.md updates

**Files:**
- Create: `docs/validation/pil-procedure.md`
- Create: `docs/validation/hil-procedure.md`
- Modify: `packages/sim/src/sim/CONTEXT.md` (correct the stale `build_tc_packet` import note at lines 23-26; append a new config-matrix section after the existing GSE-deferred note at lines 27-29)
- Create: `packages/gse/src/gse/CONTEXT.md`
- Test: `packages/sim/tests/test_validation_docs_present.py` (Create)

> Documentation task. The regression guard is a small presence/marker test asserting the four docs
> exist and carry the load-bearing "DEFINED, NOT RUN" / permanent-gap / canonical-import markers.
> The PIL/HIL docs describe how to run each profile **when hardware exists** and are explicitly
> marked DEFINED-NOT-RUN. The CONTEXT.md updates record the config-matrix axes, the `step_once`
> seam, the permanent ground-segment gap, **and** correct the now-stale `build_tc_packet` import
> note (Phase A relocated the canonical symbol from `flight.iss_iface.ingress` to
> `flight.libs.commands`; the old path is a back-compat re-export only).
>
> **CI-collection note:** the guard test lives under `packages/sim/tests/` so CI collects it.

- [ ] **Step 1: Write the failing test**
```python
# packages/sim/tests/test_validation_docs_present.py
"""Presence and marker checks for the validation procedure docs and CONTEXT updates."""

from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    """Walk up to the directory holding docs/validation/pil-procedure.md (or its parent docs/)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "docs" / "validation").exists() or (parent / "docs").exists():
            if (parent / "packages").exists():
                return parent
    raise FileNotFoundError("could not locate the repo root above the test file")


def _read(rel: str) -> str:
    """Read a repo-relative file as UTF-8 text."""
    return (_repo_root() / rel).read_text(encoding="utf-8")


def test_pil_procedure_defined_not_run() -> None:
    """PIL procedure doc exists and is marked DEFINED, NOT RUN."""
    text = _read("docs/validation/pil-procedure.md")
    assert "DEFINED, NOT RUN" in text
    assert "profiles/pil.toml" in text


def test_hil_procedure_defined_not_run() -> None:
    """HIL procedure doc exists and is marked DEFINED, NOT RUN."""
    text = _read("docs/validation/hil-procedure.md")
    assert "DEFINED, NOT RUN" in text
    assert "profiles/hil.toml" in text


def test_sim_context_mentions_matrix_and_seam() -> None:
    """sim CONTEXT.md documents the config matrix and the step_once seam."""
    text = _read("packages/sim/src/sim/CONTEXT.md")
    assert "step_once" in text
    assert "EnvironmentConfig" in text


def test_sim_context_cites_canonical_build_tc_packet_home() -> None:
    """sim CONTEXT.md cites flight.libs.commands as the canonical build_tc_packet import."""
    text = _read("packages/sim/src/sim/CONTEXT.md")
    assert "flight.libs.commands" in text
    assert "build_tc_packet" in text


def test_gse_context_present_with_permanent_gap() -> None:
    """gse CONTEXT.md exists and records the permanent ground-segment gap."""
    text = _read("packages/gse/src/gse/CONTEXT.md")
    assert "ground segment" in text.lower()
    assert "step_once" in text
```

- [ ] **Step 2: Run test to verify it fails**
Run: `uv run pytest packages/sim/tests/test_validation_docs_present.py -v`
Expected: FAIL — the PIL/HIL/gse docs do not exist (`_read` raises `FileNotFoundError`), and
`test_sim_context_mentions_matrix_and_seam` / `test_sim_context_cites_canonical_build_tc_packet_home`
fail because the existing sim CONTEXT.md still cites `flight.iss_iface.ingress` and has no
`step_once`/`EnvironmentConfig` content.

- [ ] **Step 3: Create the docs and update CONTEXT files**

First, correct the stale `build_tc_packet` import note in `packages/sim/src/sim/CONTEXT.md`. Replace
the existing bullet at lines 23-26:
```markdown
- Command-path SIL tests build signed packets with `build_tc_packet` from
  `flight.iss_iface.ingress`, pass them as `inbound_packets`, and subscribe to `CommandMsg` /
  `CommandAckMsg` on the bus to assert acceptance or rejection. The SIL test key must match
  the key used in `build_tc_packet`.
```
with:
```markdown
- Command-path SIL tests build signed packets with `build_tc_packet`, pass them as
  `inbound_packets`, and subscribe to `CommandMsg` / `CommandAckMsg` on the bus to assert
  acceptance or rejection. The SIL test key must match the key used in `build_tc_packet`.
  **Canonical import:** `from flight.libs.commands import build_tc_packet` -- the symbol was
  relocated there so the command codec lives beside the command dictionary it serializes. The old
  `flight.iss_iface.ingress` path is a back-compat re-export only; new code (SIL tests and the GSE
  `StationEmulator` alike) must import from `flight.libs.commands`.
```

Create `docs/validation/pil-procedure.md`:
```markdown
# PIL (Processor-in-the-Loop) Validation Procedure

> **STATUS: DEFINED, NOT RUN.** This procedure is specified ahead of hardware so the seam and
> profile exist and are import-clean. It is **not** executed in CI and requires a Jetson target
> plus the real compute/link/clock stack. Do not mark any requirement `verified` from PIL until
> this procedure has actually been run on hardware.

## What PIL exercises

PIL runs the flight apps with the compute, link, and clock axes set to **real** while the sensor
and gimbal remain **sim** (`profiles/pil.toml`: `sensor="sim"`, `gimbal="sim"`,
`compute="real"`, `link="real"`, `clock="real"`, `host="jetson_aarch64"`). This proves the real
ONNX detector, the real socket station link, and wall-clock timing on the target board, while
still feeding deterministic scene frames and a sim gimbal.

## Prerequisites

- Jetson aarch64 target with the lean flight image (onnxruntime + the exported model at
  `config.inference.model_path`).
- Networking between the target and a ground station emulator (`packages/gse`
  `StationEmulator`) reachable at `config.link.command_tcp_host:command_tcp_port` /
  `telemetry_udp_host:telemetry_udp_port`.
- The PIL socket harness backend (`gse.harness.SocketBackend`) -- **deferred**; it currently
  raises `NotImplementedError("PIL/HIL socket backend deferred")`. Implementing it is the
  next, human-gated effort (see the CHECKPOINT in the plan).

## Procedure (when hardware exists)

1. Flash the lean flight image to the Jetson and copy `config/default.toml` + `profiles/pil.toml`.
2. On the target, load config as an override:
   `load_config("config/default.toml", "profiles/pil.toml")`.
3. Construct drivers with `flight.core.select_drivers.select_drivers(config, RealClock(), sim_inputs)`
   where `sim_inputs` supplies frames + the scripted detector for the still-sim sensor axis
   (compute/link/clock select real branches automatically).
4. Start the real `Scheduler` (thread-per-app) -- PIL uses the real-time scheduler, not the
   deterministic stepper.
5. From the ground (`StationEmulator`), drive the realtime scenarios, including the
   `ack_within_seconds` realtime-only assertions that the in-process backend skips.
6. Record evidence against the PIL-venue requirements and update `vcrm.toml`
   (`status = "verified"`) only after a clean run.

## Notes

- The `lock` (LaunchLock) axis has no device and no config field; it remains a permanent VCRM gap
  and is not exercised by PIL.
```

Create `docs/validation/hil-procedure.md`:
```markdown
# HIL (Hardware-in-the-Loop) Validation Procedure

> **STATUS: DEFINED, NOT RUN.** This procedure is specified ahead of hardware. It is **not**
> executed in CI and requires the full flight hardware bench (camera, gimbal, radio/socket link).
> Do not mark any requirement `verified` from HIL until this procedure has actually been run.

## What HIL exercises

HIL runs every axis **real** (`profiles/hil.toml`: all five axes `"real"`,
`host="jetson_aarch64"`): the PySpin camera (`RealSensor`), the serial gimbal (`RealGimbal`,
requires `config.gimbal.serial_port` nonempty), the real ONNX detector, the socket station link
(`RealStationLink`), and `RealClock`. It is the highest-fidelity venue short of flight.

## Prerequisites

- Full bench: camera connected (PySpin SDK present), gimbal on its serial port, radio or socket
  bridge to the ground station emulator.
- The HIL socket harness backend (`gse.harness.SocketBackend`) -- **deferred** (raises
  `NotImplementedError("PIL/HIL socket backend deferred")`). Bench runners are the next,
  human-gated effort.

## Procedure (when hardware exists)

1. Provision the bench and verify each SDK loads (PySpin, pyserial, onnxruntime) -- these imports
   are lazy and only resolve when the real drivers are constructed.
2. Load config: `load_config("config/default.toml", "profiles/hil.toml")`.
3. Construct drivers with `select_drivers(config, RealClock())` (no `sim_inputs` needed -- every
   axis selects a real branch). The real sensor branch also applies
   `set_exposure_us(config.sensor.default_exposure_us)` and `set_gain_db(config.sensor.default_gain_db)`,
   exiting on `Err`.
4. Start the real `Scheduler`; drive scenarios from the ground station, including realtime-only
   assertions.
5. Record evidence against HIL-venue requirements; update `vcrm.toml` only after a clean run.

## Notes

- The `lock` (LaunchLock) axis remains a permanent VCRM gap: no device, no config field, no HIL
  coverage. It is documented, never tested.
```

Append to `packages/sim/src/sim/CONTEXT.md` (after the existing GSE-deferred bullet, line 29):
```markdown

## Config-matrix axes and the `step_once` seam (validation effort)

The validation venues are driven by `flight.libs.config.config.EnvironmentConfig` -- five
per-axis `AxisMode` knobs (`sensor`/`gimbal`/`compute`/`link`/`clock`, each `"sim"` or `"real"`)
plus `host`. `profiles/*.toml` are config **overrides** applied via
`load_config("config/default.toml", "profiles/NAME.toml")`: `sil` (all sim), `sil-link-real`
(link real, rest sim) are the **running** venues; `pil`/`hil` are **DEFINED-NOT-RUN** (see
`docs/validation/`). `flight.core.select_drivers.select_drivers(config, clock, sim_inputs)` maps
each axis to a sim or real driver; the clock axis is resolved by the composition root *before*
calling it.

The single-step body of `SilHarness.step` is extracted verbatim into
`sim.sil.stepping.step_once(...)` -- a driver-agnostic, Protocol-typed function. `SilHarness.step`
delegates to it, and the GSE `InProcessBackend` reuses the **same** `step_once` so the in-process
validation harness and the SIL harness share one stepping implementation. Do not fork a second
stepper.

The `lock` axis (LaunchLock) is intentionally absent from `EnvironmentConfig`: no device exists,
so it is a permanent VCRM gap, not a config knob.
```

Create `packages/gse/src/gse/CONTEXT.md`:
```markdown
# `gse` Subsystem Context

Non-obvious context for the ground-support-equipment package (`pact-gse`). The GSE stands in for
the real ground segment in validation; it is **test tooling**, not flight software.

---

## One-way leaf in the layer graph

`gse` may import `flight.libs` + `sim` ONLY. `flight` and `sim` must **never** import `gse`
(enforced by the `flight-gse-isolation` and `sim-gse-isolation` import-linter contracts). The
emulator is a consumer of the flight wire formats, never a dependency of the flight build.

## Builds packets from the canonical command codec

`StationEmulator.send_command` builds signed CCSDS telecommands with
`flight.libs.commands.build_tc_packet` -- the canonical home of that symbol after Phase A relocated
it there (the `flight.iss_iface.ingress` path is a back-compat re-export only). Import it from
`flight.libs.commands`, never from `flight.iss_iface.ingress`.

## The validation config matrix

The validation venues are selected by `flight.libs.config.config.EnvironmentConfig` (five
`AxisMode` axes -- `sensor`/`gimbal`/`compute`/`link`/`clock` -- plus `host`), applied as
`profiles/*.toml` overrides. Running venues: `sil` (all sim) and `sil-link-real` (link real).
DEFINED-NOT-RUN venues: `pil`, `hil` (see `docs/validation/`). `gse.orchestrator.run_scenario`
scores **frame-portable** assertions and records **realtime-only** assertions as `skip` under the
deterministic in-process backend.

## Shared stepper -- reuse, do not fork

`gse.harness.InProcessBackend` steps the flight apps via `sim.sil.stepping.step_once(...)` -- the
exact same driver-agnostic function `SilHarness.step` delegates to. The in-process backend builds
config via `load_config(default, profile)`, picks `ManualClock`, builds `SimDriverInputs` from the
scene, calls `select_drivers` + `build_apps`, and (for `link="real"`) stands up a
`StationEmulator`. `SocketBackend` (PIL/HIL) is deferred and raises `NotImplementedError`.

## Permanent gap: the real ground segment is never tested

`StationEmulator` is a stand-in. The **real ground segment** is never exercised by any running or
defined venue -- this is a permanent VCRM gap recorded in `docs/requirements/vcrm.toml`
(`GAP-GROUND-SEGMENT`). The `lock` (LaunchLock) axis is a second permanent gap: no device, no
config field.
```

- [ ] **Step 4: Run test to verify it passes**
Run: `uv run pytest packages/sim/tests/test_validation_docs_present.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Run gates**
```bash
uv run pytest packages/sim/tests/test_validation_docs_present.py -v
```
Expected: PASS. (Markdown-only changes; no ruff/mypy impact beyond the test file, which D5 covers.)

- [ ] **Step 6: Commit**
```bash
git add docs/validation/pil-procedure.md docs/validation/hil-procedure.md \
  packages/sim/src/sim/CONTEXT.md packages/gse/src/gse/CONTEXT.md \
  packages/sim/tests/test_validation_docs_present.py
git commit -m "docs(validation): PIL/HIL procedures + CONTEXT updates for the config matrix

Add DEFINED-NOT-RUN PIL/HIL run procedures; document the EnvironmentConfig
axes, the shared step_once seam, and the permanent ground-segment / LaunchLock
gaps in the sim and gse CONTEXT files. Correct the stale sim CONTEXT note to
cite flight.libs.commands.build_tc_packet as the canonical import after the
Phase A relocation (flight.iss_iface.ingress is back-compat only).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task D5: Whole-suite gate run + CHECKPOINT

**Files:**
- No code changes. This task is the final verification gate for the config-matrix effort and the
  human-gated stopping point.

> This task runs all five CI gates plus the VCRM check across the whole `packages/` tree (which now
> collects the four Phase-D guard tests, since they live under `packages/sim/tests/`) and confirms
> the entire effort (Phases A-D) is green before declaring it complete. It then records the
> CHECKPOINT: PIL/HIL plumbing (the `SocketBackend` and the Jetson/bench runners) is explicitly out
> of scope and is the next, human-gated effort.

- [ ] **Step 1: Run the full verification command block**
Run (Windows PowerShell; each must succeed):
```powershell
uv run ruff check packages scripts
uv run ruff format --check packages scripts
uv run mypy packages
uv run lint-imports
uv run python scripts/check_vcrm.py
uv run pytest packages -m "not e2e"
uv run pytest tests -m "not e2e"
```
Expected:
- `ruff check` / `ruff format --check`: no findings.
- `mypy packages`: `Success: no issues found`.
- `lint-imports`: all contracts kept, including `flight-gse-isolation` and `sim-gse-isolation`.
- `check_vcrm.py`: prints `VCRM check passed.` and exits 0.
- `pytest packages -m "not e2e"`: all pass — includes the SIL closed-loop, GSE scenario, and
  `select_drivers` tests from earlier phases, **and** the four Phase-D guards now living under
  `packages/sim/tests/` (`test_import_contracts_gse`, `test_vcrm_seed_integrity`,
  `test_check_vcrm`, `test_validation_docs_present`).
- `pytest tests -m "not e2e"`: passes (mirrors the new CI `Root guard tests` step; collects any
  repo-root guards. Exit code 5 / "no tests ran" is acceptable only if no root tests exist — but
  the Phase-D guards live under `packages/`, so this step exists to guarantee future root guards
  are covered, matching the CI step added in D3).

> If any gate fails, fix it in the owning phase's task before proceeding — D5 is a gate, not a
> place to add code. In particular: a PEP 695 `_drain[T]` occurrence is a C4 convention regression
> (fix in C4); a missing `gse` editable install is a C1 ordering failure; a `build_tc_packet`
> ImportError under gse is a Phase-A-before-C ordering failure (see the orchestration constraints
> at the top of this section).

- [ ] **Step 2: Record the CHECKPOINT**

State explicitly (in the PR description / effort log — no file write required):

> **CHECKPOINT — config-matrix effort complete; STOP HERE.**
> Delivered: the `EnvironmentConfig` five-axis matrix; `select_drivers`; the `sil` /
> `sil-link-real` running profiles and the DEFINED-NOT-RUN `pil` / `hil` profiles; the `step_once`
> seam shared by `SilHarness` and the GSE `InProcessBackend`; the `pact-gse` package
> (`StationEmulator`, scenario model, in-process backend, orchestrator) with one-way isolation
> contracts; the seeded VCRM (`vcrm.toml` + `vcrm.md`) with the permanent ground-segment gap; the
> `check_vcrm.py` CI gate; and the `Root guard tests` CI step.
> **Next (NOT in this effort; human-gated):** PIL/HIL plumbing = implement
> `gse.harness.SocketBackend` (real TCP/UDP transport) and the Jetson/bench runners, then execute
> the `docs/validation/pil-procedure.md` / `hil-procedure.md` procedures on hardware and promote
> the corresponding VCRM rows to `status = "verified"`. Do not start that until a maintainer
> green-lights the hardware bench.

- [ ] **Step 3: Commit (CHECKPOINT marker only, if an effort log exists)**

No source changes. If the effort tracks a changelog/log file, append the CHECKPOINT statement and:
```bash
git add -A
git commit -m "chore(validation): checkpoint config-matrix effort (PIL/HIL plumbing deferred)

All five gates + check_vcrm green across packages/ (incl. the four Phase-D
guards) and the root tests/ guard step. SocketBackend and the Jetson/bench
runners are the next, human-gated effort.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```
If there is no log file to touch, skip the commit — the checkpoint is recorded in the PR
description.
