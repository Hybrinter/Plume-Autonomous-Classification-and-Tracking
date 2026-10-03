"""INIT lifecycle: bounded effect executor, verification seam, app wiring.

Executor-level tests drive LifecycleExecutor directly with protocol stubs; the
app-level tests run the real INIT graph against a scripted or empty
InferenceRuntime and observe ordered effect telemetry, one INIT_COMPLETE
request, and the fail-closed OPERATE gate.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import replace

import numpy as np
import pytest
from flight.hal.drivers_sim import SimGimbal, SimIssEphemeris, SimSensor
from flight.hal.interfaces.gimbal import GimbalPosition
from flight.hal.interfaces.sensor import ImagingSensor
from flight.libs.bus import MessageBus, Subscription
from flight.libs.config import PactConfig
from flight.libs.messages import (
    FaultEventMsg,
    InferenceResultMsg,
    SystemModeActivatedMsg,
    SystemModeRequestMsg,
    TelemetryEventMsg,
)
from flight.libs.time import ManualClock
from flight.libs.types import (
    ActivationKey,
    DownlinkPriority,
    Err,
    FaultCode,
    MessageType,
    MosaicFrame,
    Ok,
    Result,
    SystemMode,
)
from flight.payload.app import PayloadApp
from flight.payload.calibration_io import build_identity_calibration
from flight.payload.gimbal.request import InhibitReference
from flight.payload.graphs import init
from flight.payload.graphs.base import (
    EffectIntent,
    EffectKind,
    EffectResult,
    EffectStatus,
    InitVerificationResult,
    InitVerificationStatus,
    TickInputs,
)
from flight.payload.graphs.parameters import GraphParameters
from flight.payload.inference import InferenceRuntime, ScriptedDetector
from flight.payload.inference.runtime import (
    RuntimeSession,
    ScriptedRuntimeFactory,
    ScriptedRuntimeSession,
)
from flight.payload.lifecycle import (
    ExactHomeArrival,
    HomeArrivalService,
    InitializationVerifier,
    LifecycleExecutor,
    LifecycleObservation,
    LifecyclePoll,
    LifecycleToken,
    PendingInitializationVerifier,
    SelfTestService,
)
from flight.payload.records import HealthSample
from flight.payload.state import PayloadState, graph_name_of
from flight.payload.tracking import EncoderSample

_EPOCH = "epoch-lifecycle-test"
_KEY = ActivationKey(epoch=_EPOCH, sequence=1)
_HOME_TARGET_RAD = math.radians(45.0)


class _MemStorage:
    """In-memory StorageWriter stub."""

    def store(
        self, item_id: str, data: bytes, priority: DownlinkPriority
    ) -> Result[str, FaultCode]:
        """Return a synthetic entry id without persisting data."""
        del data, priority
        return Ok("entry_" + item_id)


class _CountingBackend:
    """DetectorBackend spy: counts detect calls; always fails typed."""

    def __init__(self) -> None:
        self.detect_calls = 0

    def detect(self, frame: object) -> Result[InferenceResultMsg, FaultCode]:
        """Count and fail; never expected to succeed in lifecycle tests."""
        del frame
        self.detect_calls += 1
        return Err(FaultCode.INFERENCE_NAN)


class _SpySensor:
    """ImagingSensor recording stage calls; returns queued frames on acquire."""

    def __init__(self, frames: list[MosaicFrame] | None = None) -> None:
        self.calls: list[str] = []
        self._queue = list(frames) if frames is not None else [_mosaic(i) for i in range(4)]

    def set_exposure_us(self, value: float) -> Result[None, FaultCode]:
        """Record the setter call and accept."""
        self.calls.append(f"set_exposure_us:{value}")
        return Ok(None)

    def set_gain_db(self, value: float) -> Result[None, FaultCode]:
        """Record the setter call and accept."""
        self.calls.append(f"set_gain_db:{value}")
        return Ok(None)

    def start_acquisition(self) -> Result[None, FaultCode]:
        """Record the start call and accept."""
        self.calls.append("start_acquisition")
        return Ok(None)

    def stop_acquisition(self) -> Result[None, FaultCode]:
        """Record the stop call and accept."""
        self.calls.append("stop_acquisition")
        return Ok(None)

    def drain_frame(self) -> Result[None, FaultCode]:
        """Record the drain call and accept."""
        self.calls.append("drain_frame")
        return Ok(None)

    def acquire_frame(self) -> Result[MosaicFrame, FaultCode]:
        """Record the acquire call and return the next queued frame."""
        self.calls.append("acquire_frame")
        if not self._queue:
            return Err(FaultCode.CAMERA_STALL)
        return Ok(self._queue.pop(0))


def _mosaic(index: int) -> MosaicFrame:
    """One valid raw mosaic frame."""
    return MosaicFrame(
        frame_id=index + 1,
        timestamp_utc="2026-06-01T00:00:00.000Z",
        timestamp_s=float(index),
        mosaic=np.full((3, 1544, 2064), 100 + index, dtype=np.uint16),
        exposure_us=1000.0,
        gain_db=0.0,
    )


# -- observation / service stubs -------------------------------------------------


def _inputs(
    key: ActivationKey,
    now: float,
    encoder: EncoderSample | None = None,
    *,
    contained: bool = False,
    feedback_valid: bool = True,
    inhibit_confirmed: bool = True,
) -> TickInputs:
    """One TickInputs with chosen encoder/health observations."""
    return TickInputs(
        now_s=now,
        timestamp_utc="",
        activation_key=key,
        encoder=encoder,
        navigation=None,
        vision=None,
        health=HealthSample(
            feedback_valid=feedback_valid,
            inhibit_confirmed=inhibit_confirmed,
            contained=contained,
        ),
    )


def _encoder(t_s: float, angle_rad: float = 0.0, seq: int = 0) -> EncoderSample:
    """One encoder sample at the given time/angle."""
    return EncoderSample(sample_id=f"enc:{seq}", t_s=t_s, angle_rad=angle_rad)


def _observation(
    key: ActivationKey,
    now: float,
    *,
    encoder: EncoderSample | None = None,
    issued_s: float = 0.0,
    home_target_rad: float = _HOME_TARGET_RAD,
    contained: bool = False,
) -> LifecycleObservation:
    """One lifecycle observation snapshot for service calls."""
    return LifecycleObservation(
        inputs=_inputs(key, now, encoder, contained=contained),
        home_target_rad=home_target_rad,
        issued_s=issued_s,
    )


def _intent(kind: EffectKind, key: ActivationKey = _KEY) -> EffectIntent:
    """One canonical effect intent for a kind under a key."""
    return EffectIntent(activation_key=key, effect_id=kind.value, kind=kind)


def _token(key: ActivationKey = _KEY, rev: int = 1) -> LifecycleToken:
    """One lifecycle token for a key/revision."""
    return LifecycleToken(activation_key=key, control_revision=rev, containment_generation=0)


class _FixedSelftest:
    """Selftest returning a configured result each call."""

    def __init__(self, result: Result[str | None, FaultCode]) -> None:
        self._result = result
        self.calls = 0

    def check(
        self, observation: LifecycleObservation, cancel: threading.Event
    ) -> Result[str | None, FaultCode]:
        """Return the configured result."""
        del observation, cancel
        self.calls += 1
        return self._result


class _FixedHome:
    """HOME arrival returning a configured result each call."""

    def __init__(self, result: Result[str | None, FaultCode]) -> None:
        self._result = result

    def check(
        self, observation: LifecycleObservation, cancel: threading.Event
    ) -> Result[str | None, FaultCode]:
        """Return the configured result."""
        del observation, cancel
        return self._result


class _FixedVerifier:
    """Verifier popping configured results; an empty queue stays PENDING."""

    def __init__(self, results: list[Result[InitVerificationResult, FaultCode]]) -> None:
        self._results = list(results)
        self.calls = 0

    def verify(
        self, observation: LifecycleObservation, cancel: threading.Event
    ) -> Result[InitVerificationResult, FaultCode]:
        """Pop the next configured result."""
        del cancel
        self.calls += 1
        if not self._results:
            return Ok(
                InitVerificationResult(
                    activation_key=observation.inputs.activation_key,
                    status=InitVerificationStatus.PENDING,
                )
            )
        return self._results.pop(0)


class _BlockedVerifier:
    """Verifier stuck on an event, ignoring cancel like a hung call."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.released = threading.Event()
        self.calls = 0

    def verify(
        self, observation: LifecycleObservation, cancel: threading.Event
    ) -> Result[InitVerificationResult, FaultCode]:
        """Block until released; never honors the cancel flag."""
        del cancel
        self.calls += 1
        self.entered.set()
        self.released.wait()
        return Ok(
            InitVerificationResult(
                activation_key=observation.inputs.activation_key,
                status=InitVerificationStatus.PENDING,
            )
        )


class _BlockedFactory:
    """RuntimeFactory stuck on an event, ignoring cancel like a hung SDK call."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.released = threading.Event()
        self.calls = 0
        self._session = ScriptedRuntimeSession(identity="gate-session", backend=_CountingBackend())

    def load(self, cancel: threading.Event) -> Result[RuntimeSession, FaultCode]:
        """Block until released; never honors the cancel flag."""
        del cancel
        self.calls += 1
        self.entered.set()
        self.released.wait()
        return Ok(self._session)


class _CountingFactory:
    """RuntimeFactory counting load calls; always returns a scripted session."""

    def __init__(self) -> None:
        self.calls = 0

    def load(self, cancel: threading.Event) -> Result[RuntimeSession, FaultCode]:
        """Count and return a fresh scripted session."""
        del cancel
        self.calls += 1
        return Ok(ScriptedRuntimeSession(identity="counted", backend=_CountingBackend()))


class _FailFactory:
    """RuntimeFactory returning a typed load failure."""

    def __init__(self, error: FaultCode = FaultCode.MODEL_CORRUPT) -> None:
        self._error = error

    def load(self, cancel: threading.Event) -> Result[RuntimeSession, FaultCode]:
        """Fail immediately with the configured code."""
        del cancel
        return Err(self._error)


class _WarmFailSession:
    """RuntimeSession whose warm_up fails typed."""

    identity: str
    backend: _CountingBackend

    def __init__(self, identity: str, backend: _CountingBackend, error: FaultCode) -> None:
        self.identity = identity
        self.backend = backend
        self._error = error

    def warm_up(self, cancel: threading.Event) -> Result[None, FaultCode]:
        """Fail warm-up with the configured code."""
        del cancel
        return Err(self._error)


class _WarmFailFactory:
    """Factory whose session loads but fails warm-up."""

    def __init__(self, error: FaultCode = FaultCode.MODEL_CORRUPT) -> None:
        self._error = error

    def load(self, cancel: threading.Event) -> Result[RuntimeSession, FaultCode]:
        """Return a session whose warm_up fails typed."""
        del cancel
        return Ok(
            _WarmFailSession(identity="warm-fail", backend=_CountingBackend(), error=self._error)
        )


def _executor(
    runtime: InferenceRuntime,
    *,
    selftest: SelfTestService | None = None,
    home: HomeArrivalService | None = None,
    verifier: InitializationVerifier | None = None,
    deadline_s: float = 30.0,
) -> LifecycleExecutor:
    """Executor over scripted stubs with fast results by default."""
    return LifecycleExecutor(
        runtime=runtime,
        selftest=selftest or _FixedSelftest(Ok("selftest:ok")),
        home_arrival=home or _FixedHome(Ok("home:ok")),
        verifier=verifier
        or _FixedVerifier(
            [
                Ok(
                    InitVerificationResult(
                        activation_key=_KEY,
                        status=InitVerificationStatus.VERIFIED,
                        evidence_id="verify:ok",
                    )
                )
            ]
        ),
        effect_deadline_s=deadline_s,
    )


def _poll_until(
    executor: LifecycleExecutor,
    token: LifecycleToken,
    observation: LifecycleObservation,
    *,
    seconds: float = 3.0,
) -> LifecyclePoll:
    """Poll until a completion appears; return the last poll regardless."""
    deadline = time.monotonic() + seconds
    poll = executor.poll(token, observation, observation.inputs.now_s)
    while not (poll.results or poll.verification is not None or poll.install is not None):
        if time.monotonic() >= deadline:
            break
        time.sleep(0.001)
        poll = executor.poll(token, observation, observation.inputs.now_s)
    return poll


def _wait_mail(executor: LifecycleExecutor, timeout_s: float = 2.0) -> None:
    """Wait until the worker mails a completion for the next poll to consume."""
    deadline = time.monotonic() + timeout_s
    while not executor._mailbox and time.monotonic() < deadline:
        time.sleep(0.001)
    assert executor._mailbox


# -- executor-level tests ---------------------------------------------------------


def test_executor_runs_intents_in_order_and_installs_verified() -> None:
    """SELFTEST/MODEL_LOAD/HOME/VERIFY complete in order; VERIFIED installs."""
    runtime = InferenceRuntime(factory=ScriptedRuntimeFactory(_CountingBackend(), "m1"))
    executor = _executor(runtime)
    token = _token()
    obs = _observation(_KEY, now=0.0)
    try:
        results: list[EffectResult] = []
        verification: InitVerificationResult | None = None
        install: RuntimeSession | None = None
        for kind in (
            EffectKind.SELFTEST,
            EffectKind.MODEL_LOAD,
            EffectKind.HOME,
            EffectKind.VERIFY_INIT,
        ):
            executor.submit((_intent(kind),), token, now=0.0)
            poll = _poll_until(executor, token, obs)
            results.extend(poll.results)
            if poll.install is not None:
                install = poll.install
            if poll.verification is not None:
                verification = poll.verification
        assert [r.kind for r in results] == [
            EffectKind.SELFTEST,
            EffectKind.MODEL_LOAD,
            EffectKind.HOME,
        ]
        # VERIFY_INIT promotes through install + verification, not an effect result.
        assert install is not None
        assert isinstance(runtime.install_verified(install), Ok)
        assert runtime.identity == "m1"
        assert verification is not None
        assert verification.status is InitVerificationStatus.VERIFIED
    finally:
        executor.shutdown()


def test_executor_rejects_stale_token_poll() -> None:
    """A poll under a different token returns nothing and hands off nothing."""
    runtime = InferenceRuntime()
    executor = _executor(runtime)
    token_a = _token(_KEY)
    token_b = _token(ActivationKey(epoch=_EPOCH, sequence=2), rev=5)
    try:
        executor.submit((_intent(EffectKind.SELFTEST),), token_a, now=0.0)
        _poll_until(executor, token_a, _observation(_KEY, 0.0))
        poll = executor.poll(token_b, _observation(token_b.activation_key, 0.0), 0.0)
        assert poll.results == ()
        assert poll.verification is None
        assert poll.install is None
    finally:
        executor.shutdown()


def test_executor_drops_completion_after_cancel() -> None:
    """A load that completes after cancel cannot install or advance."""
    gate = _BlockedFactory()
    runtime = InferenceRuntime(factory=gate)
    executor = _executor(runtime)
    token = _token()
    try:
        executor.submit((_intent(EffectKind.MODEL_LOAD),), token, now=0.0)
        executor.poll(token, _observation(_KEY, 0.0), 0.0)
        assert gate.entered.wait(timeout=2.0)
        executor.cancel()
        gate.released.set()
        deadline = time.monotonic() + 1.0
        while executor._job is not None and time.monotonic() < deadline:
            time.sleep(0.001)
        # Cancelled generation: the completion is discarded, nothing installs.
        assert runtime.snapshot() is None
        poll = executor.poll(token, _observation(_KEY, 0.0), 0.0)
        assert poll.results == ()
        assert poll.install is None
    finally:
        gate.released.set()
        executor.shutdown()


def test_executor_worker_count_bounded_across_reentry() -> None:
    """Repeated cancel/resubmit keeps exactly one worker thread alive."""
    gate = _BlockedFactory()
    runtime = InferenceRuntime(factory=gate)
    executor = _executor(runtime)
    try:
        executor.submit((_intent(EffectKind.MODEL_LOAD),), _token(), now=0.0)
        executor.poll(_token(), _observation(_KEY, 0.0), 0.0)
        assert gate.entered.wait(timeout=2.0)
        first_worker = executor._thread
        assert first_worker is not None
        for seq in range(2, 6):
            executor.cancel()
            key = ActivationKey(epoch=_EPOCH, sequence=seq)
            executor.submit((_intent(EffectKind.MODEL_LOAD, key),), _token(key), now=0.0)
        # The old job is still blocked: no second worker was spawned and no
        # second load call was made.
        assert executor._thread is first_worker
        assert gate.calls == 1
    finally:
        gate.released.set()
        executor.shutdown()


def test_executor_shutdown_returns_with_blocked_worker() -> None:
    """shutdown() never waits on a blocked SDK load."""
    gate = _BlockedFactory()
    runtime = InferenceRuntime(factory=gate)
    executor = _executor(runtime)
    try:
        executor.submit((_intent(EffectKind.MODEL_LOAD),), _token(), now=0.0)
        executor.poll(_token(), _observation(_KEY, 0.0), 0.0)
        assert gate.entered.wait(timeout=2.0)
        start = time.monotonic()
        executor.shutdown(join_timeout_s=0.05)
        assert time.monotonic() - start < 1.0
    finally:
        gate.released.set()
        executor.shutdown()


def test_executor_deadline_fails_pending_probe() -> None:
    """An always-pending probe is failed by the issue deadline, not retried forever."""
    runtime = InferenceRuntime()
    executor = _executor(runtime, selftest=_FixedSelftest(Ok(None)), deadline_s=0.5)
    token = _token()
    try:
        executor.submit((_intent(EffectKind.SELFTEST),), token, now=0.0)
        poll = executor.poll(token, _observation(_KEY, 0.0), 1.0)
        assert len(poll.results) == 1
        result = poll.results[0]
        assert result.kind is EffectKind.SELFTEST
        assert result.status is EffectStatus.FAILED
        assert result.fault is FaultCode.GIMBAL_FAULT
    finally:
        executor.shutdown()


def test_executor_deadline_is_inclusive() -> None:
    """An intent expires exactly at issued_s + deadline; probes never re-arm."""
    runtime = InferenceRuntime()
    executor = _executor(runtime, selftest=_FixedSelftest(Ok(None)), deadline_s=30.0)
    token = _token()
    try:
        executor.submit((_intent(EffectKind.SELFTEST),), token, now=0.0)
        executor.poll(token, _observation(_KEY, 0.0), 0.0)
        _wait_mail(executor)
        # Consuming a PENDING probe result does not re-arm the issue deadline.
        poll = executor.poll(token, _observation(_KEY, 29.0), 29.0)
        assert poll.results == ()
        _wait_mail(executor)
        poll = executor.poll(token, _observation(_KEY, 30.0), 30.0)
        assert len(poll.results) == 1
        assert poll.results[0].status is EffectStatus.FAILED
        assert poll.results[0].fault is FaultCode.GIMBAL_FAULT
    finally:
        executor.shutdown()


def test_executor_pending_verifier_rearms_invocation_deadline() -> None:
    """Each returning PENDING re-arms only that verifier call's bound.

    The generation may wait far past the nominal 30 s deadline while every
    verifier invocation answers inside its own window; the deferred criterion
    is never timed out by total elapsed time.
    """
    verifier = _FixedVerifier([])
    runtime = InferenceRuntime()
    executor = _executor(runtime, verifier=verifier, deadline_s=30.0)
    token = _token()
    try:
        executor.submit((_intent(EffectKind.VERIFY_INIT),), token, now=0.0)
        executor.poll(token, _observation(_KEY, 0.0), 0.0)
        for now in (29.0, 58.0, 87.9):
            _wait_mail(executor)
            poll = executor.poll(token, _observation(_KEY, now), now)
            assert poll.results == ()
            assert poll.verification is None
            assert poll.install is None
        # ~88 s of responsive pending is fine; an unanswered call is not. At
        # exactly issued + deadline the window expires inclusively.
        _wait_mail(executor)
        assert verifier.calls >= 3
        poll = executor.poll(token, _observation(_KEY, 117.9), 117.9)
        assert len(poll.results) == 1
        assert poll.results[0].kind is EffectKind.VERIFY_INIT
        assert poll.results[0].status is EffectStatus.FAILED
        assert poll.results[0].fault is FaultCode.GIMBAL_FAULT
    finally:
        executor.shutdown()


def test_executor_blocked_verifier_fails_at_deadline() -> None:
    """A verifier call that never returns expires at its invocation bound."""
    blocked = _BlockedVerifier()
    runtime = InferenceRuntime()
    executor = _executor(runtime, verifier=blocked, deadline_s=0.5)
    token = _token()
    try:
        executor.submit((_intent(EffectKind.VERIFY_INIT),), token, now=0.0)
        executor.poll(token, _observation(_KEY, 0.0), 0.0)
        assert blocked.entered.wait(timeout=2.0)
        poll = executor.poll(token, _observation(_KEY, 0.5), 0.5)
        assert len(poll.results) == 1
        assert poll.results[0].kind is EffectKind.VERIFY_INIT
        assert poll.results[0].status is EffectStatus.FAILED
        assert poll.results[0].fault is FaultCode.GIMBAL_FAULT
    finally:
        blocked.released.set()
        executor.shutdown()


def test_executor_submit_ignores_noncanonical_intents() -> None:
    """Intents whose key or effect id disagree with the token/kind never run."""
    selftest = _FixedSelftest(Ok("selftest:ok"))
    runtime = InferenceRuntime()
    executor = _executor(runtime, selftest=selftest)
    token = _token()
    wrong_key = EffectIntent(
        activation_key=ActivationKey(epoch=_EPOCH, sequence=99),
        effect_id=EffectKind.SELFTEST.value,
        kind=EffectKind.SELFTEST,
    )
    wrong_id = EffectIntent(
        activation_key=_KEY,
        effect_id="not-a-selftest",
        kind=EffectKind.SELFTEST,
    )
    try:
        executor.submit((wrong_key, wrong_id), token, now=0.0)
        executor.poll(token, _observation(_KEY, 0.0), 0.0)
        executor.poll(token, _observation(_KEY, 0.5), 0.5)
        assert selftest.calls == 0
        assert executor._thread is None
    finally:
        executor.shutdown()


def test_executor_replayed_retired_intent_never_reruns() -> None:
    """A completed intent resubmitted under the same token cannot re-execute."""
    selftest = _FixedSelftest(Ok("selftest:ok"))
    runtime = InferenceRuntime()
    executor = _executor(runtime, selftest=selftest)
    token = _token()
    intent = _intent(EffectKind.SELFTEST)
    try:
        executor.submit((intent,), token, now=0.0)
        poll = _poll_until(executor, token, _observation(_KEY, 0.0))
        assert [r.kind for r in poll.results] == [EffectKind.SELFTEST]
        assert selftest.calls == 1
        executor.submit((intent,), token, now=1.0)
        executor.poll(token, _observation(_KEY, 1.0), 1.0)
        executor.poll(token, _observation(_KEY, 1.5), 1.5)
        assert selftest.calls == 1
    finally:
        executor.shutdown()


def test_executor_token_change_cancels_blocked_generation() -> None:
    """A new token abandons the old generation and signals the running job."""
    gate = _BlockedFactory()
    runtime = InferenceRuntime(factory=gate)
    executor = _executor(runtime)
    old_key = _KEY
    new_key = ActivationKey(epoch=_EPOCH, sequence=2)
    try:
        executor.submit((_intent(EffectKind.MODEL_LOAD),), _token(old_key), now=0.0)
        executor.poll(_token(old_key), _observation(old_key, 0.0), 0.0)
        assert gate.entered.wait(timeout=2.0)
        with executor._condition:
            old_job = executor._job
        assert old_job is not None
        new_token = _token(new_key)
        executor.submit((_intent(EffectKind.MODEL_LOAD, new_key),), new_token, now=1.0)
        assert old_job.cancel.is_set()
        gate.released.set()
        # The stuck job frees the worker without mailing; the new generation's
        # intent then runs on the same worker.
        deadline = time.monotonic() + 2.0
        while executor._job is not None and time.monotonic() < deadline:
            time.sleep(0.001)
        executor.poll(new_token, _observation(new_key, 2.0), 2.0)
        deadline = time.monotonic() + 2.0
        while gate.calls < 2 and time.monotonic() < deadline:
            time.sleep(0.001)
        assert gate.calls == 2
        _wait_mail(executor)
        poll = executor.poll(new_token, _observation(new_key, 3.0), 3.0)
        assert [r.kind for r in poll.results] == [EffectKind.MODEL_LOAD]
        assert poll.results[0].status is EffectStatus.SUCCEEDED
    finally:
        gate.released.set()
        executor.shutdown()


def test_executor_terminal_failure_abandons_generation() -> None:
    """A failed effect drops queued intents; MODEL_LOAD never runs."""
    factory = _CountingFactory()
    runtime = InferenceRuntime(factory=factory)
    executor = _executor(runtime, selftest=_FixedSelftest(Err(FaultCode.GIMBAL_FAULT)))
    token = _token()
    try:
        executor.submit(
            (_intent(EffectKind.SELFTEST), _intent(EffectKind.MODEL_LOAD)),
            token,
            now=0.0,
        )
        poll = _poll_until(executor, token, _observation(_KEY, 0.0))
        assert [r.kind for r in poll.results] == [EffectKind.SELFTEST]
        assert poll.results[0].status is EffectStatus.FAILED
        executor.poll(token, _observation(_KEY, 1.0), 1.0)
        executor.poll(token, _observation(_KEY, 2.0), 2.0)
        assert factory.calls == 0
    finally:
        executor.shutdown()


@pytest.mark.parametrize(
    "verification",
    [
        InitVerificationResult(
            activation_key=_KEY,
            status=InitVerificationStatus.VERIFIED,
            evidence_id="",
        ),
        InitVerificationResult(
            activation_key=ActivationKey(epoch=_EPOCH, sequence=99),
            status=InitVerificationStatus.VERIFIED,
            evidence_id="verify:other-key",
        ),
    ],
    ids=["blank_evidence", "wrong_key"],
)
def test_executor_verified_requires_evidence_and_key(
    verification: InitVerificationResult,
) -> None:
    """VERIFIED with blank evidence or a wrong key can never promote."""
    runtime = InferenceRuntime(factory=ScriptedRuntimeFactory(_CountingBackend(), "m1"))
    executor = _executor(runtime, verifier=_FixedVerifier([Ok(verification)]))
    token = _token()
    obs = _observation(_KEY, now=0.0)
    try:
        install: RuntimeSession | None = None
        delivered: InitVerificationResult | None = None
        for kind in (
            EffectKind.SELFTEST,
            EffectKind.MODEL_LOAD,
            EffectKind.HOME,
            EffectKind.VERIFY_INIT,
        ):
            executor.submit((_intent(kind),), token, now=0.0)
            poll = _poll_until(executor, token, obs)
            if poll.install is not None:
                install = poll.install
            if poll.verification is not None:
                delivered = poll.verification
        # The verifier responded VERIFIED, but promotion was refused: no
        # session installs and nothing is handed to the graph.
        assert install is None
        assert delivered is None
        assert runtime.snapshot() is None
    finally:
        executor.shutdown()


def test_executor_verified_needs_all_prerequisites() -> None:
    """A VERIFIED response without all prerequisite effects cannot promote."""
    runtime = InferenceRuntime(factory=ScriptedRuntimeFactory(_CountingBackend(), "m1"))
    executor = _executor(runtime)
    token = _token()
    obs = _observation(_KEY, now=0.0)
    try:
        # Skip MODEL_LOAD: SELFTEST + HOME + VERIFY only.
        install: RuntimeSession | None = None
        delivered: InitVerificationResult | None = None
        for kind in (EffectKind.SELFTEST, EffectKind.HOME, EffectKind.VERIFY_INIT):
            executor.submit((_intent(kind),), token, now=0.0)
            poll = _poll_until(executor, token, obs)
            if poll.install is not None:
                install = poll.install
            if poll.verification is not None:
                delivered = poll.verification
        assert install is None
        assert delivered is None
        assert runtime.snapshot() is None
    finally:
        executor.shutdown()


def test_executor_failed_verification_passes_to_graph() -> None:
    """An explicit FAILED verification is delivered for the graph to handle."""
    runtime = InferenceRuntime(factory=ScriptedRuntimeFactory(_CountingBackend(), "m1"))
    executor = _executor(
        runtime,
        verifier=_FixedVerifier(
            [
                Ok(
                    InitVerificationResult(
                        activation_key=_KEY,
                        status=InitVerificationStatus.FAILED,
                        evidence_id="verify:fail",
                    )
                )
            ]
        ),
    )
    token = _token()
    obs = _observation(_KEY, now=0.0)
    try:
        delivered: InitVerificationResult | None = None
        for kind in (
            EffectKind.SELFTEST,
            EffectKind.MODEL_LOAD,
            EffectKind.HOME,
            EffectKind.VERIFY_INIT,
        ):
            executor.submit((_intent(kind),), token, now=0.0)
            poll = _poll_until(executor, token, obs)
            if poll.verification is not None:
                delivered = poll.verification
        assert delivered is not None
        assert delivered.status is InitVerificationStatus.FAILED
        assert runtime.snapshot() is None
    finally:
        executor.shutdown()


def test_executor_verifier_error_fails_effect() -> None:
    """A verifier Err maps to a FAILED VERIFY_INIT effect result."""
    runtime = InferenceRuntime(factory=ScriptedRuntimeFactory(_CountingBackend(), "m1"))
    executor = _executor(runtime, verifier=_FixedVerifier([Err(FaultCode.COMMAND_INVALID)]))
    token = _token()
    try:
        executor.submit((_intent(EffectKind.VERIFY_INIT),), token, now=0.0)
        poll = _poll_until(executor, token, _observation(_KEY, 0.0))
        assert len(poll.results) == 1
        result = poll.results[0]
        assert result.kind is EffectKind.VERIFY_INIT
        assert result.status is EffectStatus.FAILED
        assert result.fault is FaultCode.COMMAND_INVALID
    finally:
        executor.shutdown()


def test_executor_pending_verifier_never_promotes() -> None:
    """The production-default PendingInitializationVerifier stays pending."""
    runtime = InferenceRuntime(factory=ScriptedRuntimeFactory(_CountingBackend(), "m1"))
    executor = LifecycleExecutor(
        runtime=runtime,
        selftest=_FixedSelftest(Ok("selftest:ok")),
        home_arrival=_FixedHome(Ok("home:ok")),
        verifier=PendingInitializationVerifier(),
        effect_deadline_s=5.0,
    )
    token = _token()
    obs = _observation(_KEY, now=0.0)
    try:
        executor.submit((_intent(EffectKind.VERIFY_INIT),), token, now=0.0)
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            poll = executor.poll(token, obs, 0.0)
            assert poll.install is None
            assert poll.verification is None
            time.sleep(0.005)
        assert runtime.snapshot() is None
    finally:
        executor.shutdown()


# -- ExactHomeArrival service ------------------------------------------------------


def _home_service() -> ExactHomeArrival:
    """The default HOME arrival probe over default params."""
    return ExactHomeArrival(GraphParameters(config=PactConfig()))


def test_home_arrival_accepts_exact_later_fresh_sample() -> None:
    """Exact home pose on a fresh sample strictly later than the intent passes."""
    service = _home_service()
    obs = _observation(
        _KEY,
        now=1.0,
        encoder=_encoder(t_s=0.999, angle_rad=_HOME_TARGET_RAD),
        issued_s=0.1,
    )
    result = service.check(obs, threading.Event())
    assert isinstance(result, Ok)
    assert result.value == "home:enc:0"


@pytest.mark.parametrize(
    ("t_s", "angle_rad", "now", "issued_s"),
    [
        (0.992, _HOME_TARGET_RAD, 1.0, 0.995),  # pre-intent sample
        (0.995, _HOME_TARGET_RAD, 1.0, 0.995),  # equal to intent: not later
        (0.999, _HOME_TARGET_RAD + 1e-6, 1.0, 0.1),  # wrong angle
        (0.999, math.nan, 1.0, 0.1),  # nonfinite
        (1.1, _HOME_TARGET_RAD, 1.0, 0.1),  # future sample
        (0.9, _HOME_TARGET_RAD, 1.0, 0.1),  # stale beyond max age
    ],
    ids=["pre_intent", "equal_intent", "wrong_angle", "nonfinite", "future", "stale"],
)
def test_home_arrival_rejects_invalid_evidence(
    t_s: float, angle_rad: float, now: float, issued_s: float
) -> None:
    """Stale, future, nonfinite, wrong-angle, or pre-intent samples stay pending."""
    service = _home_service()
    obs = _observation(
        _KEY,
        now=now,
        encoder=_encoder(t_s=t_s, angle_rad=angle_rad),
        issued_s=issued_s,
    )
    result = service.check(obs, threading.Event())
    assert isinstance(result, Ok)
    assert result.value is None


# -- app-level INIT integration ---------------------------------------------------


def _plume_detector() -> ScriptedDetector:
    """Scripted detector whose mask yields one strong above-boresight blob."""
    mask = np.zeros((1544, 2064), dtype=np.float32)
    mask[149:225, 990:1074] = 1.0
    return ScriptedDetector(mask, confidence_gate=0.55, min_blob_area_px=15)


def _build_app(
    *,
    inference: InferenceRuntime | None = None,
    selftest: SelfTestService | None = None,
    home: HomeArrivalService | None = None,
    verifier: InitializationVerifier | None = None,
    effect_deadline_s: float = 30.0,
    sensor: ImagingSensor | None = None,
    cfg: PactConfig | None = None,
) -> tuple[PayloadApp, MessageBus, SimGimbal, ManualClock]:
    """Assemble a PayloadApp over sim drivers with optional lifecycle injection."""
    base = cfg or PactConfig()
    if cfg is None:
        base = replace(
            base,
            inference=replace(base.inference, tile_rows=1, tile_cols=1),
            gimbal=replace(
                base.gimbal,
                simulation=replace(base.gimbal.simulation, encoder_noise_deg=0.0),
            ),
        )
    bus = MessageBus()
    clock = ManualClock()
    gimbal = SimGimbal(clock=clock, cfg=base.gimbal, inner_dt_s=base.controller.inner.dt_s)
    eph = SimIssEphemeris(clock=clock, cfg=base.ephemeris)
    calib = build_identity_calibration(base.sensor.height_px, base.sensor.width_px)
    app = PayloadApp.from_config(
        base,
        sensor if sensor is not None else SimSensor([]),
        gimbal,
        eph,
        inference if inference is not None else InferenceRuntime.from_scripted(_plume_detector()),
        bus,
        clock,
        calib,
        _MemStorage(),
        _EPOCH,
        selftest_service=selftest,
        home_service=home,
        verifier=verifier,
        effect_deadline_s=effect_deadline_s,
    )
    return app, bus, gimbal, clock


def _publish_mode(
    bus: MessageBus,
    mode: SystemMode,
    sequence: int,
    *,
    previous_mode: SystemMode | None = None,
) -> None:
    """Publish one authority activation."""
    bus.publish(
        SystemModeActivatedMsg(
            msg_type=MessageType.SYSTEM_MODE_ACTIVATED,
            timestamp_utc="t",
            epoch=_EPOCH,
            sequence=sequence,
            previous_mode=previous_mode,
            active_mode=mode,
            reason="test",
            request_id=None,
            recovery_authorized=False,
        )
    )


def _drive(
    app: PayloadApp,
    state: PayloadState,
    *,
    ticks: int,
    el_deg: float = 0.0,
    start_seq: int = 1,
) -> PayloadState:
    """Feed fresh encoder samples and advance one outer tick per call."""
    dt = app.params.config.controller.outer.dt_s
    now = state.last_outer_s if state.last_outer_s is not None else 0.0
    for index in range(start_seq, start_seq + ticks):
        now += dt
        app.note_gimbal_feedback(GimbalPosition(el_deg=el_deg, timestamp_s=now, sequence=index))
        state, _ = app.advance_outer(state, now)
        time.sleep(0.002)
    return state


def _drain[MsgT](sub: Subscription[MsgT]) -> list[MsgT]:
    """Drain a bus subscription into a list."""
    out: list[MsgT] = []
    while not sub.empty():
        out.append(sub.get_nowait())
    return out


def _lifecycle_kinds(telems: list[TelemetryEventMsg]) -> list[str]:
    """Ordered lifecycle_effect kinds observed on telemetry."""
    return [
        str(event.payload["kind"]) for event in telems if event.event_name == "lifecycle_effect"
    ]


def test_init_runs_ordered_effects_and_requests_init_complete() -> None:
    """SELFTEST->MODEL_LOAD->HOME->VERIFY emits one INIT_COMPLETE; INIT stays selected."""
    app, bus, _gimbal, _clock = _build_app(
        selftest=_FixedSelftest(Ok("selftest:ok")),
        home=_FixedHome(Ok("home:ok")),
        verifier=_FixedVerifier(
            [
                Ok(
                    InitVerificationResult(
                        activation_key=_KEY,
                        status=InitVerificationStatus.VERIFIED,
                        evidence_id="verify:ok",
                    )
                )
            ]
        ),
    )
    telem_sub = bus.subscribe(TelemetryEventMsg)
    request_sub = bus.subscribe(SystemModeRequestMsg)
    try:
        state = app.initial_state()
        _publish_mode(bus, SystemMode.INIT, 1)
        state = app.poll_activations(state, now=0.0)
        state = _drive(app, state, ticks=60)
    finally:
        app.lifecycle.shutdown()
    kinds = _lifecycle_kinds(_drain(telem_sub))
    assert kinds == ["selftest", "model_load", "home"]
    requests = _drain(request_sub)
    init_complete = [r for r in requests if r.requested_mode is SystemMode.IDLE]
    assert len(init_complete) == 1
    assert init_complete[0].reason == "graph_intent:init_complete"
    assert graph_name_of(state) == "init"
    assert isinstance(state.reference, InhibitReference)
    assert app.inference.identity == "scripted"


def test_init_stays_pending_with_default_verifier() -> None:
    """The default PendingInitializationVerifier never completes INIT."""
    app, bus, _gimbal, _clock = _build_app(
        selftest=_FixedSelftest(Ok("selftest:ok")),
        home=_FixedHome(Ok("home:ok")),
    )
    request_sub = bus.subscribe(SystemModeRequestMsg)
    try:
        state = app.initial_state()
        _publish_mode(bus, SystemMode.INIT, 1)
        state = app.poll_activations(state, now=0.0)
        state = _drive(app, state, ticks=30)
    finally:
        app.lifecycle.shutdown()
    requests = _drain(request_sub)
    assert not any(r.requested_mode is SystemMode.IDLE for r in requests)
    assert not any(r.requested_mode is SystemMode.SAFE for r in requests)
    assert graph_name_of(state) == "init"
    assert isinstance(state.reference, InhibitReference)


@pytest.mark.parametrize(
    ("selftest_result", "factory_kind", "verifier_kind", "home_result", "fault"),
    [
        (Err(FaultCode.GIMBAL_FAULT), None, None, None, FaultCode.GIMBAL_FAULT),
        (Ok("selftest:ok"), "fail_load", None, None, FaultCode.MODEL_CORRUPT),
        (Ok("selftest:ok"), "fail_warm", None, None, FaultCode.MODEL_CORRUPT),
        (Ok("selftest:ok"), None, "err", None, FaultCode.COMMAND_INVALID),
        (Ok("selftest:ok"), None, "failed", None, FaultCode.GIMBAL_FAULT),
        (Ok("selftest:ok"), None, None, Err(FaultCode.GIMBAL_FAULT), FaultCode.GIMBAL_FAULT),
    ],
    ids=["selftest", "model_load", "warmup", "verifier_err", "verifier_failed", "home"],
)
def test_init_failure_latches_and_requests_safe(
    selftest_result: Result[str | None, FaultCode],
    factory_kind: str | None,
    verifier_kind: str | None,
    home_result: Result[str | None, FaultCode] | None,
    fault: FaultCode,
) -> None:
    """Every lifecycle failure emits a typed fault plus one SAFE request; a
    previously verified session is preserved."""
    runtime = InferenceRuntime()
    prior = ScriptedRuntimeSession(identity="prior-model", backend=_CountingBackend())
    assert isinstance(runtime.install_verified(prior), Ok)
    if factory_kind is not None:
        factory = _FailFactory() if factory_kind == "fail_load" else _WarmFailFactory()
        runtime = InferenceRuntime(factory=factory)
        assert isinstance(runtime.install_verified(prior), Ok)

    verifier = _FixedVerifier(
        [
            Ok(
                InitVerificationResult(
                    activation_key=_KEY,
                    status=InitVerificationStatus.VERIFIED,
                    evidence_id="verify:ok",
                )
            )
        ]
    )
    if verifier_kind == "err":
        verifier = _FixedVerifier([Err(FaultCode.COMMAND_INVALID)])
    elif verifier_kind == "failed":
        verifier = _FixedVerifier(
            [
                Ok(
                    InitVerificationResult(
                        activation_key=_KEY,
                        status=InitVerificationStatus.FAILED,
                    )
                )
            ]
        )

    app, bus, _gimbal, _clock = _build_app(
        inference=runtime,
        selftest=_FixedSelftest(selftest_result),
        home=_FixedHome(home_result or Ok("home:ok")),
        verifier=verifier,
    )
    fault_sub = bus.subscribe(FaultEventMsg)
    request_sub = bus.subscribe(SystemModeRequestMsg)
    try:
        state = app.initial_state()
        _publish_mode(bus, SystemMode.INIT, 1)
        state = app.poll_activations(state, now=0.0)
        state = _drive(app, state, ticks=60)
    finally:
        app.lifecycle.shutdown()
    requests = _drain(request_sub)
    assert any(r.requested_mode is SystemMode.SAFE for r in requests)
    assert not any(r.requested_mode is SystemMode.IDLE for r in requests)
    faults = _drain(fault_sub)
    assert any(f.fault_code is fault for f in faults)
    # The previously verified runtime survives a failed reinit.
    assert app.inference.identity == "prior-model"
    assert app.inference.snapshot() is prior
    assert graph_name_of(state) == "init"
    assert isinstance(state.reference, InhibitReference)


def test_init_reentry_reruns_lifecycle() -> None:
    """A second INIT activation reruns the whole lifecycle under a new token."""
    verifier = _FixedVerifier(
        [
            Ok(
                InitVerificationResult(
                    activation_key=_KEY,
                    status=InitVerificationStatus.VERIFIED,
                    evidence_id="verify:1",
                )
            ),
            Ok(
                InitVerificationResult(
                    activation_key=ActivationKey(epoch=_EPOCH, sequence=2),
                    status=InitVerificationStatus.VERIFIED,
                    evidence_id="verify:2",
                )
            ),
        ]
    )
    app, bus, _gimbal, _clock = _build_app(
        selftest=_FixedSelftest(Ok("selftest:ok")),
        home=_FixedHome(Ok("home:ok")),
        verifier=verifier,
    )
    telem_sub = bus.subscribe(TelemetryEventMsg)
    request_sub = bus.subscribe(SystemModeRequestMsg)
    try:
        state = app.initial_state()
        _publish_mode(bus, SystemMode.INIT, 1)
        state = app.poll_activations(state, now=0.0)
        state = _drive(app, state, ticks=60)
        _publish_mode(bus, SystemMode.INIT, 2, previous_mode=SystemMode.INIT)
        state = app.poll_activations(state, now=state.last_outer_s or 100.0)
        state = _drive(app, state, ticks=60, start_seq=1000)
    finally:
        app.lifecycle.shutdown()
    kinds = _lifecycle_kinds(_drain(telem_sub))
    assert kinds.count("selftest") == 2
    assert kinds.count("model_load") == 2
    requests = _drain(request_sub)
    idles = [r for r in requests if r.requested_mode is SystemMode.IDLE]
    assert len(idles) == 2


def test_operate_unverified_runtime_fails_closed() -> None:
    """OPERATE with an empty inference runtime inhibits motion and faults once."""
    backend = _CountingBackend()
    runtime = InferenceRuntime(factory=ScriptedRuntimeFactory(backend, "m1"))
    app, bus, _gimbal, _clock = _build_app(inference=runtime)
    fault_sub = bus.subscribe(FaultEventMsg)
    request_sub = bus.subscribe(SystemModeRequestMsg)
    try:
        state = app.initial_state()
        _publish_mode(bus, SystemMode.OPERATE, 1)
        state = app.poll_activations(state, now=0.0)
        state = _drive(app, state, ticks=6)
    finally:
        app.lifecycle.shutdown()
    assert isinstance(state.reference, InhibitReference)
    faults = _drain(fault_sub)
    corrupt = [f for f in faults if f.fault_code is FaultCode.MODEL_CORRUPT]
    assert len(corrupt) == 1
    requests = _drain(request_sub)
    safes = [r for r in requests if r.requested_mode is SystemMode.SAFE]
    assert len(safes) == 1
    assert backend.detect_calls == 0


def test_operate_capture_only_policy_never_invokes_unverified_detector() -> None:
    """A capture-only OPERATE policy acquires frames while the runtime stays
    empty: no detector call, no inference publication, motion inhibited."""
    base = PactConfig()
    cfg = replace(
        base,
        inference=replace(base.inference, tile_rows=1, tile_cols=1),
        payload_policy=replace(
            base.payload_policy,
            operate=replace(base.payload_policy.operate, inference_enabled=False),
        ),
        gimbal=replace(
            base.gimbal,
            simulation=replace(base.gimbal.simulation, encoder_noise_deg=0.0),
        ),
    )
    backend = _CountingBackend()
    runtime = InferenceRuntime(factory=ScriptedRuntimeFactory(backend, "m1"))
    spy = _SpySensor()
    app, bus, _gimbal, _clock = _build_app(inference=runtime, sensor=spy, cfg=cfg)
    inf_sub = bus.subscribe(InferenceResultMsg)
    request_sub = bus.subscribe(SystemModeRequestMsg)
    try:
        state = app.initial_state()
        _publish_mode(bus, SystemMode.OPERATE, 1)
        state = app.poll_activations(state, now=0.0)
        app.capture_once(state, now=1.0)  # opportunity 1 drains
        state, outcome = app.capture_once(state, now=1.03)
        state = _drive(app, state, ticks=3)
    finally:
        app.lifecycle.shutdown()
    assert outcome.fault is None
    assert spy.calls.count("acquire_frame") == 1
    assert app.capture_shell.schedule.captured_frames == 1
    assert backend.detect_calls == 0
    assert _drain(inf_sub) == []
    assert isinstance(state.reference, InhibitReference)
    assert not any(r.requested_mode is SystemMode.SAFE for r in _drain(request_sub))


def test_inference_result_uses_runtime_identity() -> None:
    """InferenceResultMsg.model_version is the session identity, not metadata."""
    runtime = InferenceRuntime.from_scripted(_plume_detector(), identity="model-42")
    app, bus, _gimbal, _clock = _build_app(inference=runtime, sensor=_SpySensor())
    inf_sub = bus.subscribe(InferenceResultMsg)
    try:
        state = app.initial_state()
        _publish_mode(bus, SystemMode.OPERATE, 1)
        state = app.poll_activations(state, now=0.0)
        app.note_gimbal_feedback(GimbalPosition(el_deg=0.0, timestamp_s=0.5, sequence=1))
        app.capture_once(state, now=1.0)  # opportunity 1 drains
        _state, _outcome = app.capture_once(state, now=1.03)
    finally:
        app.lifecycle.shutdown()
    results = _drain(inf_sub)
    assert results
    assert results[0].model_version == "model-42"


def test_init_promotion_installs_runtime_once() -> None:
    """A verified INIT installs the candidate exactly once and bumps revision once."""
    runtime = InferenceRuntime(factory=ScriptedRuntimeFactory(_CountingBackend(), "loaded-1"))
    verifier = _FixedVerifier(
        [
            Ok(
                InitVerificationResult(
                    activation_key=_KEY,
                    status=InitVerificationStatus.VERIFIED,
                    evidence_id="verify:ok",
                )
            )
        ]
    )
    app, bus, _gimbal, _clock = _build_app(
        inference=runtime,
        selftest=_FixedSelftest(Ok("selftest:ok")),
        home=_FixedHome(Ok("home:ok")),
        verifier=verifier,
    )
    try:
        state = app.initial_state()
        _publish_mode(bus, SystemMode.INIT, 1)
        state = app.poll_activations(state, now=0.0)
        revision_at_entry = state.policy_revision
        state = _drive(app, state, ticks=60)
    finally:
        app.lifecycle.shutdown()
    # The verifier answered once; ready ticks never re-execute or reinstall.
    assert verifier.calls == 1
    installed = runtime.snapshot()
    assert installed is not None
    assert installed.identity == "loaded-1"
    assert state.policy_revision == revision_at_entry + 1


def test_init_promotion_invalidates_prior_capture_context() -> None:
    """The real INIT promotion bumps policy revision, so a capture context
    stamped before it is stale even when the reloaded identity is identical."""
    runtime = InferenceRuntime.from_scripted(_CountingBackend(), identity="same-id")
    app, bus, _gimbal, _clock = _build_app(
        inference=runtime,
        selftest=_FixedSelftest(Ok("selftest:ok")),
        home=_FixedHome(Ok("home:ok")),
        verifier=_FixedVerifier(
            [
                Ok(
                    InitVerificationResult(
                        activation_key=_KEY,
                        status=InitVerificationStatus.VERIFIED,
                        evidence_id="verify:ok",
                    )
                )
            ]
        ),
    )
    try:
        state = app.initial_state()
        _publish_mode(bus, SystemMode.INIT, 1)
        state = app.poll_activations(state, now=0.0)
        with app.state_lock:
            context = app._capture_context(state)
        assert context is not None
        assert context.model_version == "same-id"
        state = _drive(app, state, ticks=60)
    finally:
        app.lifecycle.shutdown()
    assert app._context_stale(context, state)
    session = runtime.snapshot()
    assert session is not None
    assert session.identity == "same-id"


def test_operate_unverified_capture_once_immediately() -> None:
    """Fail-closed at OPERATE entry: capture_once before the first control tick
    performs no sensor HAL calls while the runtime is unverified."""
    spy = _SpySensor()
    runtime = InferenceRuntime(factory=ScriptedRuntimeFactory(_CountingBackend(), "m1"))
    app, bus, _gimbal, _clock = _build_app(inference=runtime, sensor=spy)
    fault_sub = bus.subscribe(FaultEventMsg)
    request_sub = bus.subscribe(SystemModeRequestMsg)
    try:
        state = app.initial_state()
        _publish_mode(bus, SystemMode.OPERATE, 1)
        state = app.poll_activations(state, now=0.0)
        app.capture_once(state, now=1.0)
        app.capture_once(state, now=1.03)
    finally:
        app.lifecycle.shutdown()
    assert spy.calls == []
    assert isinstance(state.reference, InhibitReference)
    corrupt = [f for f in _drain(fault_sub) if f.fault_code is FaultCode.MODEL_CORRUPT]
    assert len(corrupt) == 1
    safes = [r for r in _drain(request_sub) if r.requested_mode is SystemMode.SAFE]
    assert len(safes) == 1


def test_init_pending_verifier_stays_ready_past_deadline() -> None:
    """A responsive default-pending verifier never times INIT out.

    With a 0.5 s effect bound, several seconds of ticks span many invocation
    windows: every verifier call returns PENDING inside its bound, so READY
    stays un-failed, no fault/SAFE/IDLE is emitted, and nothing installs.
    """
    runtime = InferenceRuntime(factory=ScriptedRuntimeFactory(_CountingBackend(), "m1"))
    app, bus, _gimbal, _clock = _build_app(
        inference=runtime,
        selftest=_FixedSelftest(Ok("selftest:ok")),
        home=_FixedHome(Ok("home:ok")),
        effect_deadline_s=0.5,
    )
    request_sub = bus.subscribe(SystemModeRequestMsg)
    fault_sub = bus.subscribe(FaultEventMsg)
    try:
        state = app.initial_state()
        _publish_mode(bus, SystemMode.INIT, 1)
        state = app.poll_activations(state, now=0.0)
        state = _drive(app, state, ticks=200)
    finally:
        app.lifecycle.shutdown()
    requests = _drain(request_sub)
    assert not any(r.requested_mode is SystemMode.SAFE for r in requests)
    assert not any(r.requested_mode is SystemMode.IDLE for r in requests)
    assert _drain(fault_sub) == []
    assert isinstance(state.graph, init.State)
    assert state.graph.node is init.InitNode.READY
    assert not state.graph.failed
    assert runtime.snapshot() is None
    assert isinstance(state.reference, InhibitReference)


def test_init_no_feedback_still_reaches_deadline_failure() -> None:
    """Lost encoder feedback cannot stall an INIT effect forever.

    The INIT graph still steps on truthful no-encoder inputs each due tick, so
    the executor's expired SELFTEST reaches the graph: typed fault, one SAFE
    request, failed INIT, and nothing installed.
    """
    runtime = InferenceRuntime(factory=ScriptedRuntimeFactory(_CountingBackend(), "m1"))
    app, bus, _gimbal, _clock = _build_app(
        inference=runtime,
        selftest=_FixedSelftest(Ok(None)),
        home=_FixedHome(Ok("home:ok")),
        effect_deadline_s=0.5,
    )
    fault_sub = bus.subscribe(FaultEventMsg)
    request_sub = bus.subscribe(SystemModeRequestMsg)
    try:
        state = app.initial_state()
        _publish_mode(bus, SystemMode.INIT, 1)
        state = app.poll_activations(state, now=0.0)
        # Submit SELFTEST under fresh feedback, then stop feeding it entirely.
        state = _drive(app, state, ticks=3)
        now = state.last_outer_s or 0.0
        state, _ = app.advance_outer(state, now=now + 0.7)
    finally:
        app.lifecycle.shutdown()
    assert isinstance(state.graph, init.State)
    assert state.graph.failed
    assert any(f.fault_code is FaultCode.GIMBAL_FAULT for f in _drain(fault_sub))
    assert any(r.requested_mode is SystemMode.SAFE for r in _drain(request_sub))
    assert runtime.snapshot() is None


def test_init_blocked_factory_safe_and_shutdown() -> None:
    """A stuck model load never stalls SAFE containment or shutdown.

    The factory ignores cancellation like a hung SDK call: the latch, the
    hardware inhibit, the SAFE graph, and a bounded shutdown must all complete
    while it is still blocked, and the late completion installs nothing.
    """
    gate = _BlockedFactory()
    runtime = InferenceRuntime(factory=gate)
    app, bus, gimbal, _clock = _build_app(inference=runtime)
    request_sub = bus.subscribe(SystemModeRequestMsg)
    try:
        state = app.initial_state()
        _publish_mode(bus, SystemMode.INIT, 1)
        state = app.poll_activations(state, now=0.0)
        # Drive until MODEL_LOAD reaches the blocked factory (ObservedSelfTest
        # succeeds: fresh encoder + confirmed inhibit).
        deadline = time.monotonic() + 5.0
        while not gate.entered.is_set() and time.monotonic() < deadline:
            state = _drive(app, state, ticks=3, el_deg=45.0)
        assert gate.entered.is_set()
        # SAFE activation while the factory is blocked: the latch must engage
        # promptly without waiting on the worker.
        _publish_mode(bus, SystemMode.SAFE, 2, previous_mode=SystemMode.INIT)
        start = time.monotonic()
        state = app.poll_activations(state, now=state.last_outer_s or 0.0)
        assert time.monotonic() - start < 2.0
        assert app.containment.local_latched
        health = gimbal.read_health()
        assert isinstance(health, Ok)
        assert health.value.inhibit_confirmed
        start = time.monotonic()
        app.lifecycle.shutdown(join_timeout_s=0.05)
        assert time.monotonic() - start < 1.0
        assert graph_name_of(state) == "safe"
        assert not any(r.requested_mode is SystemMode.IDLE for r in _drain(request_sub))
    finally:
        gate.released.set()
        app.lifecycle.shutdown()
    # Even after the stuck load returns, nothing installs into the runtime.
    join_deadline = time.monotonic() + 2.0
    while app.lifecycle._job is not None and time.monotonic() < join_deadline:
        time.sleep(0.001)
    assert app.lifecycle._job is None
    assert runtime.snapshot() is None
