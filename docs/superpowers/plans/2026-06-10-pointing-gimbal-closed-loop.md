# Pointing / Gimbal Closed-Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make pointing flight-real: a closed-loop absolute+rate gimbal HAL with encoder
feedback, boresight-relative pointing math (killing `PIXEL_TO_DEG`), the existing-but-unwired
safety gates running in the live path, encoder-based runaway detection, SAFE that actually stows
the gimbal, and the ROI crop re-enable the ingest-phase audit flagged.

**Architecture:** Per spec `docs/superpowers/specs/2026-06-09-pact-flight-final-state-design.md`
Section 5. The pure controller emits a typed `GimbalRequest` (RATE during TRACKING via LQR;
ABSOLUTE for SCAN; STOW on SAFE entry); the app shell maps requests onto the expanded
`GimbalActuator` HAL (`goto_angle`/`set_rate`/`home`/`stow`/`read_position`/`read_stow_switch`)
and publishes a `GimbalCommandMsg` telemetry record. Pointing error becomes boresight-relative
degrees via the sensor IFOV with crop back-projection; the Kalman/LQR pair operates in
error-space. `SimGimbal` gains first-order dynamics so the loop is honest in SIL; `RealGimbal`
becomes a serial PTU driver (fake-serial tested). SAFE latches in the arbiter, commanded by
`ModeChangeMsg` drained from a bus subscription, and is exited the same way.

**Tech Stack:** Python 3.12+, numpy, scipy (existing LQR/Kalman), pyserial (lazy, flight-only),
pytest, mypy --strict, ruff, import-linter. Run gates from the repo root with
`uv run <tool> packages`.

**Conventions that bind every task** (from `.claude/rules/`): 100-char lines; ASCII only; full
docstrings (summary/inputs/outputs/notes + module header; tests get one-line docstrings); numpy
dtype/shape comments at declaration sites; `Result[T, E]` for library code (never raise);
`@dataclass(frozen=True, slots=True)` for data structs; enum string values mirror member names;
module docstrings cite REQ IDs. Plan code blocks abbreviate some docstrings -- the executor
writes them in full.

**Geometry locked by this plan** (Task 7): sensor mosaic 1024x1024 (12-bit), band planes
512x512, IFOV 0.02 deg/px (FOV ~10.2 deg, matching the old 256 x 0.04 value), model input
256x256. Search mode feeds the model the full plane decimated 2x (`scale_factor=0.5`); TRACKING
mode crops a full-resolution 256x256 ROI around the Kalman-estimated target (`scale_factor=1.0`).
`backproject_pixel` (already in `preprocess/crop.py`) inverts both.

**Sign conventions locked by this plan:** image +x (column) -> +azimuth; image +y (row, downward)
-> -elevation. `boresight_error_deg` returns the angular offset OF THE TARGET from boresight; a
positive az error means the target is to the right, so the gimbal must slew +az to center it.

**Out of scope (later phases):** launch-lock interlock (mechanical phase); ground EXIT_SAFE
command routing (ISS/comms phase -- SIL publishes `ModeChangeMsg` directly); `sim.twin`
scene-gimbal coupling (the scene stays static, so SIL asserts correct command direction and
mechanism, not photometric loop closure); fault-ledger persistence (data-system phase).

**Pre-existing dirty files** (never stage): `src/pact/**`, `.idea/**`,
`.claude/settings.local.json`, `.coverage`, `bash.exe.stackdump`, `tests/integration/**`,
`tests/unit/**`, `.claude/workflows/`.

---

### Task 1: GimbalCommandMode, GIMBAL_FAULT, GimbalRequest

**Files:**
- Modify: `packages/flight/src/flight/libs/types/enums.py`
- Modify: `packages/flight/src/flight/libs/types/__init__.py` (export `GimbalCommandMode`)
- Modify: `packages/flight/src/flight/fault/policy.py` (GIMBAL_FAULT -> SAFE; update docstring)
- Create: `packages/flight/src/flight/payload/gimbal/request.py`
- Modify: `packages/flight/src/flight/payload/gimbal/__init__.py` (export `GimbalRequest`)
- Test: `packages/flight/tests/test_enums.py` (extend), `packages/flight/tests/test_fault_policy.py` (extend)
- Test: `packages/flight/tests/test_gimbal_request.py` (new)

- [x] **Step 1: Write the failing tests**

Append to `test_enums.py`:

```python
def test_gimbal_command_mode_values_mirror_names() -> None:
    """GimbalCommandMode string values must mirror member names."""
    for member in GimbalCommandMode:
        assert member.value == member.name
    assert {m.name for m in GimbalCommandMode} == {"RATE", "ABSOLUTE", "STOW", "HOME"}


def test_gimbal_fault_code_exists() -> None:
    """Driver-level gimbal failures have their own fault code."""
    assert FaultCode.GIMBAL_FAULT.value == "GIMBAL_FAULT"
```

Append to `test_fault_policy.py` (match its existing style):

```python
def test_gimbal_fault_triggers_safe() -> None:
    """A gimbal driver fault routes to SAFE (stow may be impossible; annunciate loudly)."""
    assert FaultCode.GIMBAL_FAULT in SAFE_TRIGGERING_FAULTS
```

Create `test_gimbal_request.py`:

```python
"""Tests for the GimbalRequest pure-core command value."""

from flight.libs.types import GimbalCommandMode
from flight.payload.gimbal import GimbalRequest


def test_gimbal_request_carries_mode_and_values() -> None:
    """GimbalRequest is a frozen value: mode + two axis values + reason."""
    req = GimbalRequest(
        mode=GimbalCommandMode.RATE, az_deg=1.5, el_deg=-0.5, reason="tracking_target"
    )
    assert req.mode is GimbalCommandMode.RATE
    assert req.az_deg == 1.5
```

- [x] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/flight/tests/test_enums.py packages/flight/tests/test_fault_policy.py packages/flight/tests/test_gimbal_request.py -v`
Expected: FAIL (`GimbalCommandMode`, `GIMBAL_FAULT`, `GimbalRequest` missing).

- [x] **Step 3: Implement**

In `enums.py` add to `FaultCode`: `GIMBAL_FAULT = "GIMBAL_FAULT"`, and a new enum:

```python
class GimbalCommandMode(enum.Enum):
    """How a gimbal command's axis values are interpreted.

    RATE: az/el are rates in deg/s (TRACKING). ABSOLUTE: az/el are target angles in
    degrees (SCAN, acquisition repositioning). STOW/HOME: axis values are ignored;
    the driver resolves the configured stow/home pose.
    """

    RATE = "RATE"
    ABSOLUTE = "ABSOLUTE"
    STOW = "STOW"
    HOME = "HOME"
```

In `policy.py` add `FaultCode.GIMBAL_FAULT` to `SAFE_TRIGGERING_FAULTS` and update the module
docstring's partition listing.

Create `request.py`:

```python
"""GimbalRequest: the pure controller's typed command output.

A GimbalRequest is NOT a bus message: it flows by return value from the pure control
core to the payload app shell, which maps it onto GimbalActuator HAL calls and
publishes a GimbalCommandMsg telemetry record. Keeping the pure core ignorant of the
HAL preserves the pure-core contract.

Satisfies: REQ-AIML-GIMB-001, REQ-GIMB-HIGH-001.
"""

from __future__ import annotations

# stdlib
from dataclasses import dataclass

# internal
from flight.libs.types import GimbalCommandMode


@dataclass(frozen=True, slots=True)
class GimbalRequest:
    """One gimbal command decided by the pure control core.

    Attributes:
        mode: Interpretation of the axis values (RATE deg/s, ABSOLUTE deg, STOW/HOME
            ignore them).
        az_deg: Azimuth rate (RATE) or target azimuth (ABSOLUTE); 0.0 for STOW/HOME.
        el_deg: Elevation rate (RATE) or target elevation (ABSOLUTE); 0.0 for STOW/HOME.
        reason: Human-readable reason code for telemetry/logging.
    """

    mode: GimbalCommandMode
    az_deg: float
    el_deg: float
    reason: str
```

- [x] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/flight/tests/test_enums.py packages/flight/tests/test_fault_policy.py packages/flight/tests/test_gimbal_request.py -v`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add packages/flight/src/flight/libs/types packages/flight/src/flight/fault/policy.py packages/flight/src/flight/payload/gimbal packages/flight/tests/test_enums.py packages/flight/tests/test_fault_policy.py packages/flight/tests/test_gimbal_request.py
git commit -m "feat(types): GimbalCommandMode, GIMBAL_FAULT, GimbalRequest core value"
```

---

### Task 2: GimbalConfig + runaway/strike controller fields

**Files:**
- Modify: `packages/flight/src/flight/libs/config/config.py`
- Modify: `packages/flight/src/flight/libs/config/__init__.py` (export `GimbalConfig`)
- Modify: `config/default.toml`
- Modify: `packages/flight/src/flight/core/config_loader.py`
- Test: `packages/flight/tests/test_config_defaults.py`, `packages/flight/tests/test_config_loader.py` (extend per existing patterns)

- [x] **Step 1: Write the failing tests**

Extend `test_config_loader.py` (use the existing `_DEFAULT_TOML` absolute-path style):

```python
def test_gimbal_section_loads() -> None:
    """[gimbal] TOML section maps into GimbalConfig."""
    result = load_config(_DEFAULT_TOML)
    assert isinstance(result, Ok)
    g = result.value.gimbal
    assert g.az_min_deg == -90.0
    assert g.az_max_deg == 90.0
    assert g.el_min_deg == -45.0
    assert g.el_max_deg == 45.0
    assert g.max_hw_slew_rate_deg_per_s == 10.0
    assert g.stow_el_deg == -45.0
    assert g.serial_port == ""


def test_controller_runaway_fields_load() -> None:
    """Encoder-runaway tuning fields map into ControllerConfig."""
    result = load_config(_DEFAULT_TOML)
    assert isinstance(result, Ok)
    c = result.value.controller
    assert c.runaway_rate_tolerance_deg_per_s == 1.0
    assert c.runaway_strike_count == 3
```

Extend `test_config_defaults.py` to cover the `[gimbal]` section and the two new
`[controller]` keys, following its existing defaults-vs-TOML comparison pattern.

- [x] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/flight/tests/test_config_loader.py packages/flight/tests/test_config_defaults.py -v`
Expected: FAIL (`PactConfig` has no `gimbal`).

- [x] **Step 3: Implement**

In `config.py` add (before `PactConfig`):

```python
@dataclass(frozen=True)
class GimbalConfig:
    """Configuration for the gimbal hardware envelope, poses, sim dynamics, and link."""

    az_min_deg: float = -90.0  # travel limit, azimuth minimum
    az_max_deg: float = 90.0  # travel limit, azimuth maximum
    el_min_deg: float = -45.0  # travel limit, elevation minimum
    el_max_deg: float = 45.0  # travel limit, elevation maximum
    max_hw_slew_rate_deg_per_s: float = 10.0  # hardware slew envelope (driver-enforced)
    stow_az_deg: float = 0.0  # stow pose azimuth (inside travel limits)
    stow_el_deg: float = -45.0  # stow pose elevation (inside travel limits)
    home_az_deg: float = 0.0  # home pose azimuth
    home_el_deg: float = 0.0  # home pose elevation
    sim_time_constant_s: float = 0.2  # SimGimbal first-order response time constant
    sim_encoder_noise_deg: float = 0.005  # SimGimbal encoder read noise (1-sigma)
    sim_seed: int = 0  # SimGimbal noise RNG seed (SIL determinism)
    serial_port: str = ""  # PTU serial port; "" -> RealGimbal unavailable (startup error)
    serial_baud: int = 9600  # PTU serial baud rate
    counts_per_deg: float = 77.6  # PTU encoder counts per degree (E46-class resolution)
```

Add to `ControllerConfig`:

```python
    runaway_rate_tolerance_deg_per_s: float = 1.0  # commanded-vs-encoder rate divergence limit
    runaway_strike_count: int = 3  # consecutive divergent frames before GIMBAL_RUNAWAY
```

Add to `PactConfig`: `gimbal: GimbalConfig = field(default_factory=GimbalConfig)`.

In `config/default.toml` add the two `[controller]` keys plus:

```toml
[gimbal]
az_min_deg = -90.0
az_max_deg = 90.0
el_min_deg = -45.0
el_max_deg = 45.0
max_hw_slew_rate_deg_per_s = 10.0
stow_az_deg = 0.0
stow_el_deg = -45.0
home_az_deg = 0.0
home_el_deg = 0.0
sim_time_constant_s = 0.2
sim_encoder_noise_deg = 0.005
sim_seed = 0
serial_port = ""
serial_baud = 9600
counts_per_deg = 77.6
```

In `config_loader.py` add the `gimbal` block following the exact `.get()` pattern of the other
sections, map the two new controller keys, import `GimbalConfig`, pass `gimbal=gimbal_config`.

- [x] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/flight/tests/test_config_loader.py packages/flight/tests/test_config_defaults.py -v`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add packages/flight/src/flight/libs/config config/default.toml packages/flight/src/flight/core/config_loader.py packages/flight/tests/test_config_loader.py packages/flight/tests/test_config_defaults.py
git commit -m "feat(config): GimbalConfig envelope/poses/sim/link + runaway tuning"
```

---

### Task 3: HAL expansion + SimGimbal first-order dynamics (additive)

The protocol gains the closed-loop command set; `send_command` is KEPT temporarily (removed in
Task 6) so the tree stays green. `GimbalPosition` gains an encoder timestamp.

**Files:**
- Modify: `packages/flight/src/flight/hal/interfaces/gimbal.py`
- Modify: `packages/flight/src/flight/hal/drivers_sim/gimbal.py` (full dynamics rework)
- Modify: `packages/flight/src/flight/hal/drivers_real/gimbal.py` (stub the new methods)
- Modify: `packages/sim/src/sim/sil/runner.py` (construct `SimGimbal(clock)`)
- Modify (if needed): `packages/flight/src/flight/libs/time/...` -- verify `ManualClock` exposes
  an advance method; if it does not, add `advance(dt_s: float)` with a test (read
  `packages/flight/src/flight/libs/time/` first; do not guess the API)
- Test: `packages/flight/tests/test_sim_gimbal.py` (rework), `test_hal_interfaces.py` (extend),
  `test_real_drivers.py` (extend), `test_payload_app.py`/`test_composition.py` (touch only if
  their fixtures construct `SimGimbal()` -- add the clock argument)

- [x] **Step 1: Write the failing tests** (rework `test_sim_gimbal.py`)

```python
"""Tests for SimGimbal first-order dynamics, limits, and the closed-loop HAL surface."""

from flight.hal.drivers_sim import SimGimbal
from flight.libs.config import GimbalConfig
from flight.libs.time import ManualClock
from flight.libs.types import Ok


def _gimbal(clock: ManualClock, **cfg_overrides: float) -> SimGimbal:
    return SimGimbal(clock=clock, cfg=GimbalConfig(sim_encoder_noise_deg=0.0, **cfg_overrides))


def test_goto_angle_approaches_target_with_lag() -> None:
    """An absolute command moves the gimbal toward the target, not instantly onto it."""
    clock = ManualClock()
    gimbal = _gimbal(clock)
    assert isinstance(gimbal.goto_angle(10.0, 0.0), Ok)
    clock.advance(0.1)
    mid = gimbal.read_position()
    assert isinstance(mid, Ok)
    assert 0.0 < mid.value.az_deg < 10.0
    clock.advance(30.0)
    settled = gimbal.read_position()
    assert isinstance(settled, Ok)
    assert abs(settled.value.az_deg - 10.0) < 0.1


def test_slew_rate_is_limited() -> None:
    """Motion toward a far target never exceeds the hardware slew envelope."""
    clock = ManualClock()
    gimbal = _gimbal(clock, max_hw_slew_rate_deg_per_s=10.0, sim_time_constant_s=0.001)
    gimbal.goto_angle(90.0, 0.0)
    clock.advance(1.0)
    pos = gimbal.read_position()
    assert isinstance(pos, Ok)
    assert pos.value.az_deg <= 10.0 + 1e-6


def test_set_rate_integrates_and_clamps_travel() -> None:
    """Rate commands integrate position and stop at the travel limit."""
    clock = ManualClock()
    gimbal = _gimbal(clock, az_max_deg=5.0)
    assert isinstance(gimbal.set_rate(2.0, 0.0), Ok)
    clock.advance(1.0)
    pos = gimbal.read_position()
    assert isinstance(pos, Ok)
    assert abs(pos.value.az_deg - 2.0) < 1e-6
    clock.advance(10.0)
    clamped = gimbal.read_position()
    assert isinstance(clamped, Ok)
    assert clamped.value.az_deg == 5.0


def test_stow_reaches_pose_and_sets_switch() -> None:
    """stow() drives to the configured stow pose; the switch reads True on arrival."""
    clock = ManualClock()
    gimbal = _gimbal(clock)
    assert isinstance(gimbal.stow(), Ok)
    early = gimbal.read_stow_switch()
    assert isinstance(early, Ok)
    assert early.value is False
    clock.advance(60.0)
    done = gimbal.read_stow_switch()
    assert isinstance(done, Ok)
    assert done.value is True
    pos = gimbal.read_position()
    assert isinstance(pos, Ok)
    assert abs(pos.value.el_deg - (-45.0)) < 0.5


def test_read_position_is_timestamped() -> None:
    """Encoder reads carry the monotonic read time."""
    clock = ManualClock()
    gimbal = _gimbal(clock)
    clock.advance(3.5)
    pos = gimbal.read_position()
    assert isinstance(pos, Ok)
    assert pos.value.timestamp_s == clock.monotonic_s()
```

(If `ManualClock` has a different advance API after reading the time lib, adapt the tests to it
-- the assertions stand.)

- [x] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/flight/tests/test_sim_gimbal.py -v`
Expected: FAIL (`SimGimbal.__init__` takes no clock; new methods missing).

- [x] **Step 3: Implement**

`interfaces/gimbal.py` -- `GimbalPosition` gains `timestamp_s: float` (monotonic seconds at the
encoder read); protocol gains (keep `send_command` for now, documented as deprecated pending
Task 6):

```python
    def goto_angle(self, az_deg: float, el_deg: float) -> Result[None, FaultCode]:
        """Command an absolute pointing; the driver clamps to travel limits."""
        ...

    def set_rate(
        self, az_rate_deg_per_s: float, el_rate_deg_per_s: float
    ) -> Result[None, FaultCode]:
        """Command axis rates; the driver clamps to the hardware slew envelope."""
        ...

    def home(self) -> Result[None, FaultCode]:
        """Drive to the configured home pose."""
        ...

    def stow(self) -> Result[None, FaultCode]:
        """Drive to the configured stow pose (the SAFE-mode mechanical safing action)."""
        ...

    def read_position(self) -> Result[GimbalPosition, FaultCode]:
        """Read timestamped encoder angles."""
        ...

    def read_stow_switch(self) -> Result[bool, FaultCode]:
        """Read the stow switch: True when mechanically at the stow pose."""
        ...
```

`drivers_sim/gimbal.py` -- full rework:

```python
"""Simulated gimbal with first-order dynamics, travel/slew limits, and encoder noise.

Position integrates lazily: every public call first advances the internal state by the
clock time elapsed since the previous call, so the same driver is honest under the
threaded flight loop (RealClock) and the stepped SIL (ManualClock). ABSOLUTE/STOW/HOME
approach their target with a first-order exponential response clamped to the hardware
slew envelope; RATE integrates the clamped commanded rates. Position is clamped to the
travel limits after every update. Encoder reads add seeded Gaussian noise and carry
the monotonic read timestamp. send_command (delta) is retained temporarily for the
legacy path and is removed by the pointing switchover.

Satisfies: REQ-AIML-GIMB-001, REQ-GIMB-HIGH-002.
"""

from __future__ import annotations

# stdlib
import math

# third-party
import numpy as np

# internal
from flight.hal.interfaces.gimbal import GimbalPosition
from flight.libs.config import GimbalConfig
from flight.libs.messages import GimbalCommandMsg
from flight.libs.time import Clock
from flight.libs.types import FaultCode, GimbalCommandMode, Ok, Result

_STOW_TOLERANCE_DEG = 0.5  # switch closes within this of the stow pose


class SimGimbal:
    """Gimbal driver with first-order dynamics for SIL (satisfies GimbalActuator)."""

    def __init__(
        self,
        clock: Clock,
        cfg: GimbalConfig | None = None,
        az_deg: float = 0.0,
        el_deg: float = 0.0,
    ) -> None:
        """Start at a pose with the configured dynamics and a seeded noise RNG."""
        self._clock = clock
        self._cfg = cfg if cfg is not None else GimbalConfig()
        self._az = az_deg
        self._el = el_deg
        self._mode: GimbalCommandMode | None = None
        self._target_az = az_deg
        self._target_el = el_deg
        self._rate_az = 0.0
        self._rate_el = 0.0
        self._stow_commanded = False
        self._last_t = clock.monotonic_s()
        self._rng = np.random.default_rng(self._cfg.sim_seed)

    def _clamp_travel(self) -> None:
        """Clamp the integrated pose into the configured travel limits."""
        cfg = self._cfg
        self._az = min(max(self._az, cfg.az_min_deg), cfg.az_max_deg)
        self._el = min(max(self._el, cfg.el_min_deg), cfg.el_max_deg)

    def _integrate(self) -> None:
        """Advance the pose by the clock time elapsed since the last call."""
        now = self._clock.monotonic_s()
        dt = now - self._last_t
        self._last_t = now
        if dt <= 0.0:
            return
        cfg = self._cfg
        max_step = cfg.max_hw_slew_rate_deg_per_s * dt
        if self._mode is GimbalCommandMode.RATE:
            self._az += min(max(self._rate_az * dt, -max_step), max_step)
            self._el += min(max(self._rate_el * dt, -max_step), max_step)
        elif self._mode is not None:
            alpha = 1.0 - math.exp(-dt / cfg.sim_time_constant_s)
            for axis in ("az", "el"):
                pos = getattr(self, f"_{axis}")
                err = getattr(self, f"_target_{axis}") - pos
                step = min(max(err * alpha, -max_step), max_step)
                setattr(self, f"_{axis}", pos + step)
        self._clamp_travel()

    def goto_angle(self, az_deg: float, el_deg: float) -> Result[None, FaultCode]:
        """Set an absolute target, clamped into the travel limits."""
        self._integrate()
        cfg = self._cfg
        self._target_az = min(max(az_deg, cfg.az_min_deg), cfg.az_max_deg)
        self._target_el = min(max(el_deg, cfg.el_min_deg), cfg.el_max_deg)
        self._mode = GimbalCommandMode.ABSOLUTE
        self._stow_commanded = False
        return Ok(None)

    def set_rate(
        self, az_rate_deg_per_s: float, el_rate_deg_per_s: float
    ) -> Result[None, FaultCode]:
        """Set axis rates, clamped to the hardware slew envelope."""
        self._integrate()
        limit = self._cfg.max_hw_slew_rate_deg_per_s
        self._rate_az = min(max(az_rate_deg_per_s, -limit), limit)
        self._rate_el = min(max(el_rate_deg_per_s, -limit), limit)
        self._mode = GimbalCommandMode.RATE
        self._stow_commanded = False
        return Ok(None)

    def home(self) -> Result[None, FaultCode]:
        """Drive to the configured home pose."""
        self._integrate()
        self._target_az, self._target_el = self._cfg.home_az_deg, self._cfg.home_el_deg
        self._mode = GimbalCommandMode.HOME
        self._stow_commanded = False
        return Ok(None)

    def stow(self) -> Result[None, FaultCode]:
        """Drive to the configured stow pose and arm the stow switch."""
        self._integrate()
        self._target_az, self._target_el = self._cfg.stow_az_deg, self._cfg.stow_el_deg
        self._mode = GimbalCommandMode.STOW
        self._stow_commanded = True
        return Ok(None)

    def read_position(self) -> Result[GimbalPosition, FaultCode]:
        """Return the noisy, timestamped encoder pose."""
        self._integrate()
        noise = self._rng.normal(0.0, self._cfg.sim_encoder_noise_deg, 2)
        return Ok(
            GimbalPosition(
                az_deg=self._az + float(noise[0]),
                el_deg=self._el + float(noise[1]),
                timestamp_s=self._last_t,
            )
        )

    def read_stow_switch(self) -> Result[bool, FaultCode]:
        """True once stow was commanded and the pose is within the switch tolerance."""
        self._integrate()
        at_pose = (
            abs(self._az - self._cfg.stow_az_deg) < _STOW_TOLERANCE_DEG
            and abs(self._el - self._cfg.stow_el_deg) < _STOW_TOLERANCE_DEG
        )
        return Ok(self._stow_commanded and at_pose)

    def send_command(self, command: GimbalCommandMsg) -> Result[None, FaultCode]:
        """DEPRECATED legacy delta path (removed by the pointing switchover)."""
        self._integrate()
        self._az += command.az_delta_deg
        self._el += command.el_delta_deg
        self._clamp_travel()
        return Ok(None)
```

`drivers_real/gimbal.py` -- add the new methods as stubs (`goto_angle`/`set_rate`/`home`/`stow`
return `Ok(None)`; `read_position` returns the origin with `timestamp_s=0.0`;
`read_stow_switch` returns `Ok(False)`), full implementation in Task 8.

`sim/sil/runner.py` -- `SimGimbal(clock, cfg=config.gimbal)`. Update any other
`SimGimbal()`/`GimbalPosition(...)` constructions in tests to the new signatures.

- [x] **Step 4: Run the full gates**

Run: `uv run pytest packages` and `uv run mypy packages`
Expected: PASS (legacy delta path still wired; new surface tested).

- [x] **Step 5: Commit**

```bash
git add packages/flight/src/flight/hal packages/flight/src/flight/libs/time packages/sim/src/sim/sil/runner.py packages/flight/tests
git commit -m "feat(hal): closed-loop gimbal surface + SimGimbal first-order dynamics"
```

---

### Task 4: Boresight-relative pointing math

**Files:**
- Create: `packages/flight/src/flight/payload/gimbal/pointing.py`
- Modify: `packages/flight/src/flight/payload/gimbal/__init__.py` (export)
- Test: `packages/flight/tests/test_pointing.py`

- [x] **Step 1: Write the failing tests**

```python
"""Tests for boresight-relative pointing geometry."""

from flight.payload.gimbal import boresight_error_deg, target_displacement_px


def test_centered_target_has_zero_error() -> None:
    """A centroid at the plane center yields zero pointing error (the silent-wrongness fix)."""
    az, el = boresight_error_deg(
        centroid_px=(256.0, 256.0),
        crop_origin_px=(0, 0),
        scale_factor=1.0,
        plane_width_px=512,
        plane_height_px=512,
        ifov_deg_per_px=0.02,
    )
    assert az == 0.0
    assert el == 0.0


def test_offsets_map_through_ifov_with_image_sign_convention() -> None:
    """+x offset -> +az; +y (downward) offset -> -el; scaled by IFOV."""
    az, el = boresight_error_deg(
        centroid_px=(306.0, 206.0),
        crop_origin_px=(0, 0),
        scale_factor=1.0,
        plane_width_px=512,
        plane_height_px=512,
        ifov_deg_per_px=0.02,
    )
    assert abs(az - 1.0) < 1e-9  # (306-256) * 0.02
    assert abs(el - 1.0) < 1e-9  # -(206-256) * 0.02


def test_crop_and_scale_backproject_before_conversion() -> None:
    """Crop origin and decimation scale are inverted before the angular conversion."""
    az, el = boresight_error_deg(
        centroid_px=(85.0, 85.0),
        crop_origin_px=(0, 0),
        scale_factor=0.5,  # decimated search mode: tensor px = plane px * 0.5
        plane_width_px=512,
        plane_height_px=512,
        ifov_deg_per_px=0.02,
    )
    assert abs(az - (170.0 - 256.0) * 0.02) < 1e-9
    assert abs(el - (-(170.0 - 256.0) * 0.02)) < 1e-9


def test_displacement_is_full_frame_euclidean_pixels() -> None:
    """Deadband displacement is measured in full-frame plane pixels."""
    d = target_displacement_px(
        centroid_px=(85.0, 85.0),
        crop_origin_px=(0, 0),
        scale_factor=0.5,
        plane_width_px=512,
        plane_height_px=512,
    )
    expected = (2.0 * (170.0 - 256.0) ** 2) ** 0.5
    assert abs(d - expected) < 1e-9
```

- [x] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/flight/tests/test_pointing.py -v`
Expected: FAIL (module missing).

- [x] **Step 3: Implement `pointing.py`**

```python
"""Boresight-relative pointing geometry: tensor pixels -> angular error.

Replaces the absolute-centroid * PIXEL_TO_DEG bug (baseline Section 4.4 of the parity
baseline): error is measured FROM THE PLANE CENTER (boresight), after inverting the
preprocess crop/decimation transform, and converted to degrees via the sensor IFOV.
Sign convention: image +x (column) -> +azimuth; image +y (row, downward) -> -elevation.
The returned error is the target's angular offset from boresight -- the slew needed to
center it has the same sign.

Satisfies: REQ-AIML-GIMB-002, REQ-GIMB-HIGH-001.
"""

from __future__ import annotations

# stdlib
import math


def _full_frame_px(
    centroid_px: tuple[float, float],
    crop_origin_px: tuple[int, int],
    scale_factor: float,
) -> tuple[float, float]:
    """Invert the crop/scale transform: tensor pixel -> full-plane pixel (float)."""
    return (
        crop_origin_px[0] + centroid_px[0] / scale_factor,
        crop_origin_px[1] + centroid_px[1] / scale_factor,
    )


def boresight_error_deg(
    centroid_px: tuple[float, float],
    crop_origin_px: tuple[int, int],
    scale_factor: float,
    plane_width_px: int,
    plane_height_px: int,
    ifov_deg_per_px: float,
) -> tuple[float, float]:
    """Angular (az, el) offset of a detected centroid from the boresight, in degrees.

    Inputs: tensor-space centroid, the ProcessedFrameMsg crop_origin_px/scale_factor
    that produced the tensor, the band-plane geometry, and the per-pixel IFOV.
    Output: (az_error_deg, el_error_deg) per the module sign convention.
    """
    full_x, full_y = _full_frame_px(centroid_px, crop_origin_px, scale_factor)
    az_err = (full_x - plane_width_px / 2.0) * ifov_deg_per_px
    el_err = -(full_y - plane_height_px / 2.0) * ifov_deg_per_px
    return (az_err, el_err)


def target_displacement_px(
    centroid_px: tuple[float, float],
    crop_origin_px: tuple[int, int],
    scale_factor: float,
    plane_width_px: int,
    plane_height_px: int,
) -> float:
    """Euclidean full-plane pixel distance of the centroid from boresight (deadband input)."""
    full_x, full_y = _full_frame_px(centroid_px, crop_origin_px, scale_factor)
    return math.hypot(full_x - plane_width_px / 2.0, full_y - plane_height_px / 2.0)
```

- [x] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/flight/tests/test_pointing.py -v` -- Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add packages/flight/src/flight/payload/gimbal packages/flight/tests/test_pointing.py
git commit -m "feat(pointing): boresight-relative error via IFOV with crop backprojection"
```

---

### Task 5: Encoder-based runaway monitor

**Files:**
- Create: `packages/flight/src/flight/payload/gimbal/runaway.py`
- Modify: `packages/flight/src/flight/payload/gimbal/__init__.py` (export)
- Test: `packages/flight/tests/test_runaway.py`

- [x] **Step 1: Write the failing tests**

```python
"""Tests for the encoder-divergence runaway monitor."""

from flight.hal.interfaces import GimbalPosition
from flight.libs.types import FaultCode
from flight.payload.gimbal import RunawayState, check_runaway


def _pos(az: float, el: float, t: float) -> GimbalPosition:
    return GimbalPosition(az_deg=az, el_deg=el, timestamp_s=t)


def test_matching_motion_resets_strikes() -> None:
    """Encoder motion matching the commanded rate keeps the strike count at zero."""
    state = RunawayState(last_pos=_pos(0.0, 0.0, 0.0), strike_count=2)
    new_state, fault = check_runaway(
        state, _pos(2.0, 0.0, 1.0), 2.0, 0.0, True, tolerance_deg_per_s=1.0, strike_limit=3
    )
    assert fault is None
    assert new_state.strike_count == 0


def test_divergence_accumulates_strikes_then_faults() -> None:
    """Sustained commanded-vs-encoder divergence raises GIMBAL_RUNAWAY at the strike limit."""
    state = RunawayState(last_pos=_pos(0.0, 0.0, 0.0), strike_count=0)
    for i in range(1, 3):
        state, fault = check_runaway(
            state,
            _pos(0.0, 0.0, float(i)),  # gimbal not moving
            2.0,  # but commanded 2 deg/s az
            0.0,
            True,
            tolerance_deg_per_s=1.0,
            strike_limit=3,
        )
        assert fault is None
        assert state.strike_count == i
    state, fault = check_runaway(
        state, _pos(0.0, 0.0, 3.0), 2.0, 0.0, True, tolerance_deg_per_s=1.0, strike_limit=3
    )
    assert fault is FaultCode.GIMBAL_RUNAWAY


def test_no_rate_mode_or_missing_data_resets() -> None:
    """Outside RATE mode, or without a prior/current read, the monitor resets quietly."""
    state = RunawayState(last_pos=_pos(0.0, 0.0, 0.0), strike_count=2)
    new_state, fault = check_runaway(
        state, _pos(0.0, 0.0, 1.0), 2.0, 0.0, False, tolerance_deg_per_s=1.0, strike_limit=3
    )
    assert fault is None
    assert new_state.strike_count == 0
    new_state, fault = check_runaway(
        new_state, None, 2.0, 0.0, True, tolerance_deg_per_s=1.0, strike_limit=3
    )
    assert fault is None
    assert new_state.last_pos is None
```

- [x] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/flight/tests/test_runaway.py -v` -- Expected: FAIL.

- [x] **Step 3: Implement `runaway.py`**

```python
"""Encoder-divergence runaway monitor (pure).

Replaces pixel-inferred runaway with physics: while the controller is commanding
rates (RATE mode), the measured encoder rate between consecutive reads must agree
with the commanded rate within a tolerance. Sustained divergence over strike_limit
consecutive checks raises GIMBAL_RUNAWAY (motor stall, encoder fault, or actuation
without authority). Outside RATE mode -- or when either read is missing or time does
not advance -- the monitor resets rather than guessing (ABSOLUTE/STOW/HOME approach
profiles are driver-internal, so the expected rate is unknown).

Satisfies: REQ-AIML-GIMB-007, REQ-GIMB-HIGH-003.
"""

from __future__ import annotations

# stdlib
import math
from dataclasses import dataclass

# internal
from flight.hal.interfaces import GimbalPosition
from flight.libs.types import FaultCode


@dataclass(frozen=True, slots=True)
class RunawayState:
    """Monitor state threaded across frames.

    Attributes:
        last_pos: The previous encoder read, or None before the first read.
        strike_count: Consecutive divergent checks so far.
    """

    last_pos: GimbalPosition | None
    strike_count: int


INITIAL_RUNAWAY_STATE = RunawayState(last_pos=None, strike_count=0)


def check_runaway(
    state: RunawayState,
    pos: GimbalPosition | None,
    commanded_az_rate_deg_per_s: float,
    commanded_el_rate_deg_per_s: float,
    rate_mode_active: bool,
    tolerance_deg_per_s: float,
    strike_limit: int,
) -> tuple[RunawayState, FaultCode | None]:
    """Compare measured encoder rate against the commanded rate; strike on divergence.

    Returns (new_state, GIMBAL_RUNAWAY | None); the fault fires when strike_count
    reaches strike_limit.
    """
    if pos is None:
        return (RunawayState(last_pos=None, strike_count=0), None)
    if (
        not rate_mode_active
        or state.last_pos is None
        or pos.timestamp_s <= state.last_pos.timestamp_s
    ):
        return (RunawayState(last_pos=pos, strike_count=0), None)
    dt = pos.timestamp_s - state.last_pos.timestamp_s
    actual_az = (pos.az_deg - state.last_pos.az_deg) / dt
    actual_el = (pos.el_deg - state.last_pos.el_deg) / dt
    divergence = math.hypot(
        actual_az - commanded_az_rate_deg_per_s, actual_el - commanded_el_rate_deg_per_s
    )
    strikes = state.strike_count + 1 if divergence > tolerance_deg_per_s else 0
    fault = FaultCode.GIMBAL_RUNAWAY if strikes >= strike_limit else None
    return (RunawayState(last_pos=pos, strike_count=strikes), fault)
```

- [x] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/flight/tests/test_runaway.py -v` -- Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add packages/flight/src/flight/payload/gimbal packages/flight/tests/test_runaway.py
git commit -m "feat(gimbal): encoder-divergence runaway monitor"
```

---

### Task 6: The pointing switchover (arbiter, control, app, HAL, messages)

The one coordinated commit: the controller emits `GimbalRequest`, SAFE latches and stows, the
safety gates run live, `PIXEL_TO_DEG` dies, and the legacy delta path is removed. Everything
nontrivial was unit-tested in Tasks 1-5. Work the files in order; run the full gates at the end.

**Files:**
- Modify: `packages/flight/src/flight/libs/messages/messages.py` (GimbalCommandMsg reshape;
  InferenceResultMsg gains crop fields)
- Modify: `packages/flight/src/flight/payload/model/detector.py` (+ `blobs.py` only if needed)
  (copy crop fields into InferenceResultMsg in both backends)
- Modify: `packages/flight/src/flight/payload/gimbal/arbiter.py` (rework)
- Modify: `packages/flight/src/flight/payload/gimbal/__init__.py`
- Modify: `packages/flight/src/flight/payload/control.py` (rework)
- Modify: `packages/flight/src/flight/payload/app.py` (mode subscription + request mapping)
- Modify: `packages/flight/src/flight/hal/interfaces/gimbal.py` (delete `send_command`)
- Modify: `packages/flight/src/flight/hal/drivers_sim/gimbal.py`,
  `packages/flight/src/flight/hal/drivers_real/gimbal.py` (delete `send_command`)
- Modify: `packages/flight/src/flight/core/composition.py` (PayloadController needs SensorConfig
  -- only if from_config signatures change there)
- Modify: `packages/sim/src/sim/sil/runner.py` (harness drains modes + passes gimbal position)
- Test (rework): `test_arbiter.py`, `test_payload_controller.py`, `test_payload_app.py`,
  `test_messages.py`, `test_scripted_detector.py`, `test_onnx_detector.py`,
  `test_hal_interfaces.py`, `test_sim_gimbal.py` (drop send_command test),
  `packages/sim/tests/test_sil_closed_loop.py`

- [x] **Step 1: Messages.** `GimbalCommandMsg` becomes the telemetry record of an issued
request (delta fields die):

```python
@dataclass(frozen=True)
class GimbalCommandMsg:
    """Telemetry record of a gimbal command issued by the payload app."""

    msg_type: MessageType  # must be MessageType.GIMBAL_COMMAND
    timestamp_utc: str  # ISO 8601, millisecond precision
    frame_id: int  # frame that triggered this command
    mode: GimbalCommandMode  # RATE / ABSOLUTE / STOW / HOME
    az_value_deg: float  # rate (deg/s) for RATE; target angle (deg) for ABSOLUTE; 0 otherwise
    el_value_deg: float  # rate (deg/s) for RATE; target angle (deg) for ABSOLUTE; 0 otherwise
    state: GimbalState  # arbiter state at time of command
    reason: str  # human-readable reason code for logging
```

`InferenceResultMsg` gains `crop_origin_px: tuple[int, int]` and `scale_factor: float`
(documented: the preprocess transform that produced the tensor the blobs live in). Both
detector backends copy them from the `ProcessedFrameMsg`:
`crop_origin_px=frame.crop_origin_px, scale_factor=frame.scale_factor`.

- [x] **Step 2: Arbiter rework** (`arbiter.py`). Delete `PIXEL_TO_DEG`. `ArbiterState` gains
`scan_direction: float = 1.0` and `miss_count: int = 0` (docstring updated: miss_count is the
TRACKING release-hysteresis counter). New step signature and behavior:

```python
    def step(
        self,
        state: ArbiterState,
        result: InferenceResultMsg,
        error_deg: tuple[float, float] | None,
        now: float,
        safe_commanded: bool,
        safe_cleared: bool,
    ) -> tuple[ArbiterState, GimbalRequest | None, list[TelemetryEventMsg]]:
```

Behavior (each bullet is implemented exactly; transitions still emit the existing
`state_transition` TelemetryEventMsg):

- **SAFE entry**: if (`safe_commanded` or `result.mode_flags != 0`) and not already SAFE ->
  transition to SAFE and return `GimbalRequest(mode=GimbalCommandMode.STOW, az_deg=0.0,
  el_deg=0.0, reason="safe_entry_stow")`. SAFE latches: while in SAFE, no other requests are
  ever produced, blobs are ignored.
- **SAFE exit**: in SAFE and `safe_cleared` -> transition to IDLE (no request); all counters
  reset (`miss_count=0`, `idle_duration_s=0.0`, `current_target_id=None`).
- **TRACKING release hysteresis**: in TRACKING with no blobs -> `miss_count += 1`; transition
  to IDLE only when `miss_count >= cfg.release_persistence_frames`; any blob resets
  `miss_count` to 0. (Replaces the immediate TRACKING->IDLE drop.)
- **TRACKING command**: when TRACKING with blobs and `error_deg` is not None and the rate
  limiter (`_rate_ok`, unchanged) allows: proportional fallback rates
  `az_rate = clip(error_deg[0] * 1.0, +-cfg.max_slew_rate_deg_per_s)` (gain 1.0 per second,
  documented; the LQR in control.py refines this when the estimator is initialized), and return
  `GimbalRequest(RATE, az_rate, el_rate, "tracking_target")`.
- **SCAN raster, absolute**: `scan_pan += scan_direction * cfg.scan_slew_rate_deg_per_s *
  (1.0 / cfg.retarget_rate_limit_hz)`; on crossing +30.0 / -30.0, clamp and flip
  `scan_direction`; return `GimbalRequest(ABSOLUTE, scan_pan, 0.0, "nadir_scan")` (fixes the
  old never-reversing delta scan).
- IDLE/ACQUIRING transition logic is otherwise unchanged.

Rework `test_arbiter.py` to the new signature; add tests for: SAFE entry emits exactly one STOW
request and latches; safe_cleared returns to IDLE; release hysteresis holds TRACKING for
`release_persistence_frames - 1` misses; SCAN reverses direction at the +30/-30 boundary;
TRACKING emits RATE requests with the proportional clip.

- [x] **Step 3: Control rework** (`control.py`). Delete `PIXEL_TO_DEG`. `ControlState` gains
`runaway: RunawayState`, `deadband_strikes: int`, `commanded_az_rate_deg_per_s: float`,
`commanded_el_rate_deg_per_s: float`. `PayloadController` gains `plane_width_px: int`,
`plane_height_px: int`, `ifov_deg_per_px: float`;
`from_config(cfg: ControllerConfig, sensor: SensorConfig)` fills them
(`plane = sensor.{width,height}_px // 2`). New step:

```python
    def step(
        self,
        state: ControlState,
        result: InferenceResultMsg,
        now: float,
        gimbal_pos: GimbalPosition | None,
        safe_commanded: bool,
        safe_cleared: bool,
    ) -> tuple[ControlState, GimbalRequest | None, list[TelemetryEventMsg], FaultCode | None]:
```

Pipeline (replaces the body; gates finally run live):

```python
        cfg = self.cfg
        gated = apply_confidence_gate(result.blobs, cfg.confidence_gate)
        gated = apply_min_area_gate(gated, cfg.min_blob_area_px)
        matched = match_blobs(
            state.arbiter.tracked_blobs, tuple(gated), cfg.blob_iou_match_threshold
        )

        error_deg: tuple[float, float] | None = None
        displacement = None
        if matched:
            error_deg = boresight_error_deg(
                matched[0].centroid_raw, result.crop_origin_px, result.scale_factor,
                self.plane_width_px, self.plane_height_px, self.ifov_deg_per_px,
            )
            displacement = target_displacement_px(
                matched[0].centroid_raw, result.crop_origin_px, result.scale_factor,
                self.plane_width_px, self.plane_height_px,
            )

        if error_deg is not None:
            ema = ema_update(state.ema, error_deg, cfg.ema_alpha)
        else:
            ema = EmaFilterState(centroid=(0.0, 0.0), initialized=False)

        kalman = predict(self.kf, state.kalman)
        if ema.initialized:
            obs = np.array([ema.centroid[0], ema.centroid[1]], dtype=np.float64)
            updated = update(self.kf, kalman, obs)
            if isinstance(updated, Ok):
                kalman = updated.value

        # Deadband + max-displacement strike gate (finally wired; REQ-AIML-GIMB-006/007).
        fault: FaultCode | None = None
        deadband_strikes = 0
        suppress_rate_command = False
        if displacement is not None:
            db = check_deadband(displacement, cfg.min_deadband_px, cfg.max_deadband_px)
            if isinstance(db, Err):
                deadband_strikes = state.deadband_strikes + 1
                suppress_rate_command = True
                if deadband_strikes >= cfg.max_deadband_strike_count:
                    fault = FaultCode.GIMBAL_RUNAWAY
            elif not db.value:
                suppress_rate_command = True

        filtered = replace(result, blobs=matched)
        new_arbiter, request, telemetry = self.arbiter.step(
            state.arbiter, filtered, error_deg, now, safe_commanded, safe_cleared
        )

        if (
            request is not None
            and request.mode is GimbalCommandMode.RATE
            and ema.initialized
        ):
            u = compute_control(self.lqr, np.asarray(kalman.x, dtype=np.float64))
            limit = cfg.max_slew_rate_deg_per_s
            request = replace(
                request,
                az_deg=float(min(max(u[0], -limit), limit)),
                el_deg=float(min(max(u[1], -limit), limit)),
            )
        if request is not None and request.mode is GimbalCommandMode.RATE and (
            suppress_rate_command
        ):
            request = None

        cmd_az, cmd_el = state.commanded_az_rate_deg_per_s, state.commanded_el_rate_deg_per_s
        rate_mode_active = cmd_az != 0.0 or cmd_el != 0.0
        new_runaway, runaway_fault = check_runaway(
            state.runaway, gimbal_pos, cmd_az, cmd_el, rate_mode_active,
            cfg.runaway_rate_tolerance_deg_per_s, cfg.runaway_strike_count,
        )
        if fault is None:
            fault = runaway_fault

        next_cmd_az = request.az_deg if request and request.mode is GimbalCommandMode.RATE else 0.0
        next_cmd_el = request.el_deg if request and request.mode is GimbalCommandMode.RATE else 0.0
        new_state = ControlState(
            arbiter=new_arbiter, ema=ema, kalman=kalman, runaway=new_runaway,
            deadband_strikes=deadband_strikes,
            commanded_az_rate_deg_per_s=next_cmd_az, commanded_el_rate_deg_per_s=next_cmd_el,
        )
        return new_state, request, telemetry, fault
```

`initial_state()` adds `runaway=INITIAL_RUNAWAY_STATE, deadband_strikes=0,
commanded_az_rate_deg_per_s=0.0, commanded_el_rate_deg_per_s=0.0`. Notes in the docstring: the
EMA/Kalman now live in boresight-error degree space (the LQR setpoint "target at boresight" is
the zero vector, so `u = -K x` needs no explicit subtraction); STOW/ABSOLUTE requests are never
deadband-suppressed (safing and scan must always actuate).

Rework `test_payload_controller.py`; add tests: deadband suppression below `min_deadband_px`;
strike-count fault above `max_deadband_px` for `max_deadband_strike_count` frames; runaway fault
propagates from a stalled encoder while rates are commanded; SAFE entry produces the STOW
request and a latched state.

- [x] **Step 4: App rework** (`app.py`). `PayloadApp` gains `mode_sub:
Subscription[ModeChangeMsg]` (created in `from_config` via `bus.subscribe(ModeChangeMsg)`).
New helper:

```python
    def poll_mode_changes(self) -> tuple[bool, bool]:
        """Drain pending ModeChangeMsg; return (safe_commanded, safe_cleared).

        SAFE requests latch the payload via the arbiter; any non-SAFE mode message is
        the ground-commanded recovery signal. Both may be True in one drain (last
        writer wins downstream: the arbiter applies safe_commanded first).
        """
        safe_commanded = False
        safe_cleared = False
        while not self.mode_sub.empty():
            msg = self.mode_sub.get_nowait()
            if msg.new_mode is SystemMode.SAFE:
                safe_commanded = True
            else:
                safe_cleared = True
        return safe_commanded, safe_cleared
```

`process_frame(raw, state, now, slew_rate_deg_per_s=0.0, gimbal_pos=None,
safe_commanded=False, safe_cleared=False)`. After detection, the control call and actuation
become:

```python
        new_state, request, telemetry, ctrl_fault = self.controller.step(
            state, inference, now, gimbal_pos, safe_commanded, safe_cleared
        )
        for event in telemetry:
            self.bus.publish(event)
        if ctrl_fault is not None:
            self._publish_fault(ctrl_fault, f"control fault frame_id={raw.frame_id}")

        if request is not None:
            if request.mode is GimbalCommandMode.RATE:
                send_result = self.gimbal.set_rate(request.az_deg, request.el_deg)
            elif request.mode is GimbalCommandMode.ABSOLUTE:
                send_result = self.gimbal.goto_angle(request.az_deg, request.el_deg)
            elif request.mode is GimbalCommandMode.STOW:
                send_result = self.gimbal.stow()
            else:
                send_result = self.gimbal.home()
            if isinstance(send_result, Err):
                self._publish_fault(
                    send_result.error, f"gimbal actuation failed frame_id={raw.frame_id}"
                )
            self.bus.publish(
                GimbalCommandMsg(
                    msg_type=MessageType.GIMBAL_COMMAND,
                    timestamp_utc=self.clock.wall_clock_iso(),
                    frame_id=raw.frame_id,
                    mode=request.mode,
                    az_value_deg=request.az_deg,
                    el_value_deg=request.el_deg,
                    state=new_state.arbiter.gimbal_state,
                    reason=request.reason,
                )
            )
```

`run()`: each iteration calls `safe_commanded, safe_cleared = self.poll_mode_changes()` before
acquiring; passes `gimbal_pos=pos_res.value if isinstance(pos_res, Ok) else None` plus the safe
flags into `process_frame` (the existing slew-rate read already provides `pos_res`). Shell-level
safety fallback (documented in the docstring): if `safe_commanded` is True and the frame
acquisition failed, call `self.gimbal.stow()` directly -- a stalled camera must not prevent
safing.

`TickOutcome.command_issued` stays (request is not None).

- [x] **Step 5: HAL cleanup.** Delete `send_command` from the `GimbalActuator` protocol and
both drivers (and its `GimbalCommandMsg` imports). The protocol docstring now states drivers
enforce the hardware envelope and the arbiter enforces the mission envelope (defense in depth).

- [x] **Step 6: SIL runner/harness.** `SilHarness.step(now)`: before processing the frame,
`safe_commanded, safe_cleared = apps.payload.poll_mode_changes()`; read
`pos = system.gimbal.read_position()`; call `process_frame(acquired.value,
self._payload_state, now, 0.0, pos.value if isinstance(pos, Ok) else None, safe_commanded,
safe_cleared)`. `run_steps` must also advance the shared `ManualClock` by `dt` each step (use
the advance API confirmed in Task 3) so SimGimbal dynamics integrate.

- [x] **Step 7: Test sweep.** Rework the listed test files to the new signatures/shapes. The
SIL nominal closed-loop test's "gimbal moved off origin" assertion now requires TRACKING to
issue RATE commands and SimGimbal to integrate them across steps (positions change between
steps). Add to `test_payload_app.py`: a `ModeChangeMsg(SAFE)` published on the bus causes the
next processed frame to issue a STOW actuation (assert via SimGimbal `read_stow_switch` after
advancing the clock, or assert the published `GimbalCommandMsg.mode is STOW`).
`rg "PIXEL_TO_DEG|az_delta_deg|send_command" packages` must return no live hits.

- [x] **Step 8: Run the full gates**

Run: `uv run pytest packages; uv run ruff check packages; uv run ruff format --check packages; uv run mypy packages; uv run lint-imports`
Expected: all green.

- [x] **Step 9: Commit**

```bash
git add packages/flight/src packages/sim/src packages/flight/tests packages/sim/tests
git commit -m "feat(pointing)!: boresight-error closed loop, SAFE stow, wired safety gates"
```

---

### Task 7: ROI crop re-enable + geometry bump (1024 mosaic)

Closes the ingest-phase audit deferral. Search mode decimates the full plane; TRACKING crops a
full-resolution ROI around the Kalman-estimated target.

**Files:**
- Modify: `packages/flight/src/flight/libs/config/config.py` + `config/default.toml`
  (SensorConfig: `width_px=1024`, `height_px=1024`, `ifov_deg_per_px=0.02`)
- Modify: `packages/flight/src/flight/payload/app.py` (from_config validation + roi selection
  in process_frame)
- Modify: `packages/sim/src/sim/scene/plume.py` (1024 mosaic; plume off-center)
- Test: `packages/flight/tests/test_payload_app.py` (extend), `packages/flight/tests/test_config_defaults.py`,
  `packages/sim/tests/test_scene.py`, `packages/sim/tests/test_sil_closed_loop.py`

- [ ] **Step 1: Write the failing tests**

In `test_payload_app.py` (using the existing app fixture pattern, sensor now 1024):

```python
def test_search_mode_decimates_full_plane() -> None:
    """Outside TRACKING the model sees the decimated full plane (scale 0.5)."""
    # process one frame from IDLE; capture the published InferenceResultMsg
    # (subscribe InferenceResultMsg on the test bus before processing):
    assert msg.scale_factor == 0.5
    assert msg.crop_origin_px == (0, 0)


def test_tracking_mode_crops_full_resolution_roi() -> None:
    """In TRACKING with an initialized estimator, a 256x256 scale-1.0 ROI is cropped."""
    # drive the app to TRACKING (ScriptedDetector blobs persist >= acquire frames),
    # then process one more frame and capture the InferenceResultMsg:
    assert msg.scale_factor == 1.0
    assert msg.crop_origin_px != (0, 0)
```

In `test_scene.py`: mosaic shape becomes `(1024, 1024)`; the plume brightness check moves to the
region around plane (340, 340) (see Step 3 geometry).

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/flight/tests/test_payload_app.py packages/sim/tests/test_scene.py -v`
Expected: FAIL (sensor still 512; no ROI logic).

- [ ] **Step 3: Implement**

Config: `SensorConfig.width_px = 1024`, `height_px = 1024`, `ifov_deg_per_px = 0.02` (+ TOML;
field comment notes FOV parity with the previous 512 x 0.04 geometry).

`app.py` `from_config` validation replaces the strict equality:

```python
        plane_h, plane_w = cfg.sensor.height_px // 2, cfg.sensor.width_px // 2
        if plane_h < cfg.inference.input_height_px or plane_w < cfg.inference.input_width_px:
            raise ValueError("band plane must be at least the inference input size")
        if (
            plane_h % cfg.inference.input_height_px
            or plane_w % cfg.inference.input_width_px
            or plane_h // cfg.inference.input_height_px != plane_w // cfg.inference.input_width_px
        ):
            raise ValueError("plane size must be an integer multiple of the inference input")
```

`process_frame`, after `normalize_dn`/`select_bands` produce the full `(4, plane, plane)`
stack, replace the fixed `crop_origin_px=(0, 0), scale_factor=1.0` with mode-dependent ROI:

```python
        plane_h, plane_w = selected.value.shape[1], selected.value.shape[2]
        in_tracking = (
            state.arbiter.gimbal_state is GimbalState.TRACKING and state.ema.initialized
        )
        if in_tracking:
            # Full-resolution ROI centered on the Kalman-estimated target.
            est_az = float(state.kalman.x[0])
            est_el = float(state.kalman.x[1])
            center_x = int(plane_w / 2 + est_az / self.sensor_cfg.ifov_deg_per_px)
            center_y = int(plane_h / 2 - est_el / self.sensor_cfg.ifov_deg_per_px)
            tensor, crop_origin = crop_to_roi(
                selected.value,
                (center_x, center_y),
                (self.inference_cfg.input_height_px, self.inference_cfg.input_width_px),
            )
            scale = 1.0
        else:
            # Decimated full-plane search mode.
            factor = plane_h // self.inference_cfg.input_height_px
            tensor = selected.value[:, ::factor, ::factor]
            crop_origin = (0, 0)
            scale = 1.0 / factor
```

(`crop_to_roi` imported from `flight.payload.preprocess`; quality flags run on the full
`selected.value` before cropping, unchanged.) The `ProcessedFrameMsg` gets
`tensor=tensor, crop_origin_px=crop_origin, scale_factor=scale`.

`plume.py`: `FRAME_SIZE = 1024` (planes 512); `_PLUME_CENTER = (340.0, 340.0)` in band-plane px
(tensor (170, 170) decimated -> full-plane displacement ~119 px from boresight (512/2=256):
above `min_deadband_px=20`, below `max_deadband_px=250`, so TRACKING commands flow);
`_PLUME_SIGMA = 24.0`. The `plume_detector()` scripted mask square moves to
`mask[145:195, 145:195]` (tensor-space center ~170, matching the scene in decimated search
mode). Update SIL closed-loop assertions accordingly.

- [ ] **Step 4: Run the full gates**

Run: `uv run pytest packages` and `uv run mypy packages` -- Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/flight/src packages/sim/src config packages/flight/tests packages/sim/tests
git commit -m "feat(payload): ROI crop in TRACKING + decimated search mode (1024 mosaic)"
```

---

### Task 8: RealGimbal -- serial PTU driver

**Files:**
- Modify: `packages/flight/src/flight/hal/drivers_real/gimbal.py` (full implementation)
- Modify: `packages/flight/src/flight/core/main.py` (construct from config; fail startup
  without a port)
- Test: `packages/flight/tests/test_real_gimbal_serial.py` (new, fake-serial)
- Test: `packages/flight/tests/test_real_drivers.py` (keep/adjust construction tests)

- [ ] **Step 1: Write the failing tests** -- fake `serial` module injected via
`monkeypatch.setitem(sys.modules, "serial", fake)` (same pattern as the fake-PySpin tests in
`test_real_sensor_pyspin.py` -- read that file first and mirror its structure):

```python
"""RealGimbal behavior tests against a fake pyserial module (no SDK in CI)."""

import sys
import types

import pytest

from flight.libs.config import GimbalConfig
from flight.libs.time import ManualClock
from flight.libs.types import Err, FaultCode, Ok


class _FakeSerial:
    """Scriptable serial port: records writes, replays queued response lines."""

    def __init__(self, port: str, baudrate: int, timeout: float) -> None:
        self.writes: list[bytes] = []
        self.responses: list[bytes] = []

    def write(self, data: bytes) -> int:
        self.writes.append(data)
        return len(data)

    def readline(self) -> bytes:
        return self.responses.pop(0) if self.responses else b""


def _install_fake_serial(monkeypatch: pytest.MonkeyPatch) -> type[_FakeSerial]:
    fake = types.ModuleType("serial")

    class SerialException(Exception):
        pass

    fake.Serial = _FakeSerial  # type: ignore[attr-defined]
    fake.SerialException = SerialException  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "serial", fake)
    return _FakeSerial


def test_goto_angle_writes_position_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    """goto_angle converts degrees to encoder counts and writes PP/TP commands."""
    _install_fake_serial(monkeypatch)
    from flight.hal.drivers_real import RealGimbal

    gimbal = RealGimbal(clock=ManualClock(), cfg=GimbalConfig(serial_port="COM3"))
    gimbal._port.responses = [b"*\n", b"*\n"]
    result = gimbal.goto_angle(10.0, -5.0)
    assert isinstance(result, Ok)
    assert gimbal._port.writes[0] == b"PP776\n"  # 10.0 deg * 77.6 counts/deg
    assert gimbal._port.writes[1] == b"TP-388\n"


def test_goto_angle_clamps_to_travel_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Targets outside the travel envelope are clamped before conversion."""
    _install_fake_serial(monkeypatch)
    from flight.hal.drivers_real import RealGimbal

    gimbal = RealGimbal(clock=ManualClock(), cfg=GimbalConfig(serial_port="COM3"))
    gimbal._port.responses = [b"*\n", b"*\n"]
    assert isinstance(gimbal.goto_angle(500.0, 0.0), Ok)
    assert gimbal._port.writes[0] == b"PP6984\n"  # clamped to az_max 90 deg


def test_error_response_is_gimbal_fault(monkeypatch: pytest.MonkeyPatch) -> None:
    """A '!' response from the PTU maps to Err(GIMBAL_FAULT)."""
    _install_fake_serial(monkeypatch)
    from flight.hal.drivers_real import RealGimbal

    gimbal = RealGimbal(clock=ManualClock(), cfg=GimbalConfig(serial_port="COM3"))
    gimbal._port.responses = [b"! illegal command\n"]
    result = gimbal.goto_angle(1.0, 0.0)
    assert isinstance(result, Err)
    assert result.error == FaultCode.GIMBAL_FAULT


def test_read_position_parses_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    """read_position queries PP/TP and converts counts back to timestamped degrees."""
    _install_fake_serial(monkeypatch)
    from flight.hal.drivers_real import RealGimbal

    clock = ManualClock()
    gimbal = RealGimbal(clock=clock, cfg=GimbalConfig(serial_port="COM3"))
    gimbal._port.responses = [b"* 776\n", b"* -388\n"]
    result = gimbal.read_position()
    assert isinstance(result, Ok)
    assert abs(result.value.az_deg - 10.0) < 1e-6
    assert abs(result.value.el_deg - (-5.0)) < 1e-6
    assert result.value.timestamp_s == clock.monotonic_s()


def test_missing_pyserial_raises_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without pyserial installed, constructing RealGimbal raises ImportError."""
    monkeypatch.setitem(sys.modules, "serial", None)
    from flight.hal.drivers_real import RealGimbal

    with pytest.raises(ImportError):
        RealGimbal(clock=ManualClock(), cfg=GimbalConfig(serial_port="COM3"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/flight/tests/test_real_gimbal_serial.py -v`
Expected: FAIL (constructor signature; behavior missing).

- [ ] **Step 3: Implement RealGimbal**

```python
"""Real two-axis PTU gimbal driver over a serial ASCII protocol (reference: FLIR PTU
E46-class; spec Section 2).

pyserial imports lazily in __init__ (the SDK-free CI pattern). Protocol subset, line
oriented: commands are '<verb><signed counts>\n' writes; every command yields one
response line -- '*' prefix = success, '!' prefix = error. Angle <-> count conversion
uses GimbalConfig.counts_per_deg. Verbs: PP/TP = pan/tilt absolute position command
or (bare) position query; PS/TS = pan/tilt rate. The exact verb set is a documented
reference assumption to be validated at HIL bring-up against the actual unit's manual.
The driver enforces the travel and slew envelopes by clamping before conversion
(defense in depth below the arbiter's mission limits). A lock serializes transactions
(capture loop vs control plane). Driver-level failures map to GIMBAL_FAULT.

Satisfies: REQ-AIML-GIMB-001, REQ-GIMB-HIGH-004.
"""

from __future__ import annotations

# stdlib
import threading

# internal
from flight.hal.interfaces.gimbal import GimbalPosition
from flight.libs.config import GimbalConfig
from flight.libs.time import Clock
from flight.libs.types import Err, FaultCode, Ok, Result


class RealGimbal:
    """Serial PTU driver satisfying GimbalActuator structurally."""

    def __init__(
        self,
        clock: Clock,
        cfg: GimbalConfig | None = None,
        timeout_s: float = 1.0,
    ) -> None:
        """Open the configured serial port.

        Raises:
            ImportError: If pyserial is not installed.
            ValueError: If cfg.serial_port is empty (startup misconfiguration).
        """
        try:
            import serial
        except ImportError as exc:
            raise ImportError(
                "pyserial is not installed. Install it to use RealGimbal; use "
                "SimGimbal in tests and simulation."
            ) from exc
        self._cfg = cfg if cfg is not None else GimbalConfig()
        if not self._cfg.serial_port:
            raise ValueError("GimbalConfig.serial_port must be set to use RealGimbal")
        self._serial_exc = serial.SerialException
        self._port = serial.Serial(
            port=self._cfg.serial_port, baudrate=self._cfg.serial_baud, timeout=timeout_s
        )
        self._clock = clock
        self._lock = threading.Lock()

    def _transact(self, command: str) -> Result[str, FaultCode]:
        """Write one command line and read its response; '!' or I/O error -> GIMBAL_FAULT."""
        try:
            self._port.write(f"{command}\n".encode("ascii"))
            response = self._port.readline().decode("ascii", errors="replace").strip()
        except self._serial_exc:
            return Err(FaultCode.GIMBAL_FAULT)
        if not response.startswith("*"):
            return Err(FaultCode.GIMBAL_FAULT)
        return Ok(response)

    def _counts(self, deg: float) -> int:
        """Convert degrees to encoder counts."""
        return round(deg * self._cfg.counts_per_deg)

    def goto_angle(self, az_deg: float, el_deg: float) -> Result[None, FaultCode]:
        """Command absolute pan/tilt positions, clamped to the travel envelope."""
        cfg = self._cfg
        az = min(max(az_deg, cfg.az_min_deg), cfg.az_max_deg)
        el = min(max(el_deg, cfg.el_min_deg), cfg.el_max_deg)
        with self._lock:
            for verb, value in (("PP", az), ("TP", el)):
                result = self._transact(f"{verb}{self._counts(value)}")
                if isinstance(result, Err):
                    return Err(result.error)
        return Ok(None)

    def set_rate(
        self, az_rate_deg_per_s: float, el_rate_deg_per_s: float
    ) -> Result[None, FaultCode]:
        """Command pan/tilt rates, clamped to the hardware slew envelope."""
        limit = self._cfg.max_hw_slew_rate_deg_per_s
        az = min(max(az_rate_deg_per_s, -limit), limit)
        el = min(max(el_rate_deg_per_s, -limit), limit)
        with self._lock:
            for verb, value in (("PS", az), ("TS", el)):
                result = self._transact(f"{verb}{self._counts(value)}")
                if isinstance(result, Err):
                    return Err(result.error)
        return Ok(None)

    def home(self) -> Result[None, FaultCode]:
        """Drive to the configured home pose."""
        return self.goto_angle(self._cfg.home_az_deg, self._cfg.home_el_deg)

    def stow(self) -> Result[None, FaultCode]:
        """Drive to the configured stow pose."""
        return self.goto_angle(self._cfg.stow_az_deg, self._cfg.stow_el_deg)

    def read_position(self) -> Result[GimbalPosition, FaultCode]:
        """Query pan/tilt positions and convert counts to timestamped degrees."""
        with self._lock:
            counts: list[int] = []
            for verb in ("PP", "TP"):
                result = self._transact(verb)
                if isinstance(result, Err):
                    return Err(result.error)
                try:
                    counts.append(int(result.value.lstrip("* ").strip()))
                except ValueError:
                    return Err(FaultCode.GIMBAL_FAULT)
        return Ok(
            GimbalPosition(
                az_deg=counts[0] / self._cfg.counts_per_deg,
                el_deg=counts[1] / self._cfg.counts_per_deg,
                timestamp_s=self._clock.monotonic_s(),
            )
        )

    def read_stow_switch(self) -> Result[bool, FaultCode]:
        """Infer stow from encoder pose (the reference PTU exposes no discrete switch)."""
        pos = self.read_position()
        if isinstance(pos, Err):
            return Err(pos.error)
        return Ok(
            abs(pos.value.az_deg - self._cfg.stow_az_deg) < 0.5
            and abs(pos.value.el_deg - self._cfg.stow_el_deg) < 0.5
        )
```

`main.py` `build_flight_system`: `gimbal=RealGimbal(clock=clock, cfg=config.gimbal)`; the
empty-port `ValueError` propagates as the startup failure. Adjust `test_real_drivers.py`
constructions.

- [ ] **Step 4: Run the full gates**

Run: `uv run pytest packages` and `uv run mypy packages` -- Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/flight/src/flight/hal/drivers_real/gimbal.py packages/flight/src/flight/core/main.py packages/flight/tests
git commit -m "feat(hal): serial PTU RealGimbal with envelope clamps + fake-serial tests"
```

---

### Task 9: SIL closed-loop strengthening (SAFE stow, recovery, command direction)

**Files:**
- Test: `packages/sim/tests/test_sil_closed_loop.py` (extend)
- Modify (only if the tests demand it): `packages/sim/src/sim/sil/runner.py`

- [ ] **Step 1: Write the failing tests** (build on the existing closed-loop fixtures)

```python
def test_thermal_safe_stows_the_gimbal() -> None:
    """THERMAL_OVER_LIMIT -> FDIR SAFE -> arbiter STOW -> SimGimbal reaches the stow pose."""
    # existing thermal-over-limit fixture (95 C), run enough steps for FDIR to route
    # SAFE and the dynamics to settle (e.g. harness.run_steps(15, dt=1.0)):
    switch = system.gimbal.read_stow_switch()
    assert isinstance(switch, Ok)
    assert switch.value is True


def test_safe_recovery_returns_to_operations() -> None:
    """A ground ModeChangeMsg(non-SAFE) after SAFE un-latches the arbiter."""
    # drive into SAFE as above, then publish the recovery message and step once:
    system.bus.publish(
        ModeChangeMsg(
            msg_type=MessageType.MODE_CHANGE,
            timestamp_utc="2026-06-10T00:00:00.000Z",
            new_mode=SystemMode.IDLE,
            requested_by="test_ground_recovery",
        )
    )
    harness.run_steps(2, dt=1.0)
    # the arbiter must have left SAFE (it will re-acquire the scripted plume):
    assert harness.payload_gimbal_state() is not GimbalState.SAFE


def test_tracking_commands_point_toward_the_plume() -> None:
    """RATE commands during TRACKING have the sign of the boresight error (plume at
    plane (340, 340): +x +az error, +y -> -el error) and the gimbal moves that way."""
    # nominal fixture; run past acquisition into TRACKING, then:
    pos = system.gimbal.read_position()
    assert isinstance(pos, Ok)
    assert pos.value.az_deg > 0.5  # plume to the right of boresight
    assert pos.value.el_deg < -0.5  # plume below boresight (image +y)
```

(Expose `payload_gimbal_state()` on `SilHarness` -- a one-line accessor returning
`self._payload_state.arbiter.gimbal_state` -- if it does not already exist.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/sim/tests/test_sil_closed_loop.py -v`
Expected: the new tests FAIL only if Tasks 6-7 left gaps -- this task is the phase's
integration proof. Diagnose any failure to its root cause (harness wiring vs control logic);
do not weaken assertions.

- [ ] **Step 3: Make them pass** (harness accessor + any wiring fixes), run the full gates

Run: `uv run pytest packages; uv run mypy packages` -- Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add packages/sim
git commit -m "test(sil): SAFE stow, ground recovery, and command-direction closed-loop proof"
```

---

### Task 10: ADR + context docs + final verification

**Files:**
- Create: `docs/adr/NNNN-closed-loop-gimbal-pointing.md` (next number -- list `docs/adr/` first)
- Modify: `packages/flight/src/flight/payload/CONTEXT.md`,
  `packages/flight/src/flight/hal/CONTEXT.md`, `packages/flight/src/flight/libs/CONTEXT.md`,
  `packages/sim/src/sim/CONTEXT.md`, `packages/flight/src/flight/fault/CONTEXT.md`

- [ ] **Step 1: Write the ADR** -- context (open-loop delta commands, absolute-centroid error,
unwired gates, SAFE no-op: baseline Sections 4.2/5); decision (GimbalRequest core value +
absolute/rate/stow HAL; boresight-relative error via IFOV + crop backprojection; error-space
Kalman/LQR with zero setpoint; driver hardware envelope + arbiter mission envelope; encoder
runaway; latched SAFE with arbiter-issued stow + shell fallback; ROI crop in TRACKING /
decimated search); consequences (SIL exercises real dynamics; PIXEL_TO_DEG and the delta
command model are gone; scan raster is absolute and reversing; recovery requires an explicit
non-SAFE ModeChangeMsg). Reference spec Sections 5 and the 2026-06-06 baseline.

- [ ] **Step 2: Update the five CONTEXT.md files** -- payload (request flow, error-space
estimators, ROI modes, gate wiring); hal (closed-loop surface, dynamics sim, fake-serial
pattern, envelope ownership); libs (GimbalCommandMode, reshaped GimbalCommandMsg,
InferenceResultMsg crop fields); sim (ManualClock advancing, stow assertions); fault
(GIMBAL_FAULT routing).

- [ ] **Step 3: Run the full gates one final time**

Run: `uv run pytest packages; uv run ruff check packages; uv run ruff format --check packages; uv run mypy packages; uv run lint-imports`
Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add docs/adr packages
git commit -m "docs: ADR + subsystem context for closed-loop gimbal pointing"
```

---

## Self-review notes (already applied)

- Spec Section 5 coverage: closed-loop HAL command set (T3/T6), driver+arbiter limit layering
  (T3/T6/T8), SimGimbal dynamics (T3), RealGimbal serial driver (T8), boresight-relative error +
  IFOV intrinsics + setpoint LQR rate commands (T4/T6), deadband/slew/hysteresis/strike wiring
  (T6), encoder runaway (T5/T6), SAFE stow + latch + ground recovery (T6/T9), ROI crop deferral
  closed (T7). PIXEL_TO_DEG removal verified by grep in T6 Step 7.
- Type consistency: `GimbalRequest(mode, az_deg, el_deg, reason)`;
  `GimbalPosition(az_deg, el_deg, timestamp_s)`;
  `controller.step(state, result, now, gimbal_pos, safe_commanded, safe_cleared) ->
  (ControlState, GimbalRequest | None, list[TelemetryEventMsg], FaultCode | None)`;
  `arbiter.step(state, result, error_deg, now, safe_commanded, safe_cleared)`;
  `check_runaway(state, pos, cmd_az, cmd_el, rate_mode_active, tolerance, strike_limit)` --
  used identically in Tasks 4-9 code and tests.
- Known intentional coupling: Task 6 is the coordinated switchover (controller output type, HAL
  surface, and message shapes cannot change independently); all its nontrivial logic is
  unit-tested in Tasks 1-5. Task 7 isolates the geometry bump so Task 6's diff stays reviewable.
- Honesty constraints recorded: the SIL scene is static (no twin), so closed-loop tests assert
  command direction and mechanism, not photometric convergence; the PTU ASCII verb set is a
  documented reference assumption pending HIL bring-up.
