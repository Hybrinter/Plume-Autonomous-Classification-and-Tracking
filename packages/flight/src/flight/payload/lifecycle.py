"""INIT lifecycle execution: bounded effect worker and verification seam.

The INIT graph emits activation-scoped EffectIntents (SELFTEST, MODEL_LOAD,
HOME, VERIFY_INIT) and stays pure; this module is the shell-side machinery
that executes them. LifecycleToken pins every intent to the exact activation
and shell revisions that issued it; LifecycleObservation is the immutable
snapshot handed to services -- a TickInputs view plus the configured home
target, prerequisite evidence, and the warm candidate identity. Services are
Protocol-typed: the observed self-test attests only current hardware
observations (fresh valid encoder plus inhibit-confirmed, not contained), the
HOME service attests exact configured-pose arrival on a fresh strictly-later
encoder sample, and the production InitializationVerifier stays PENDING --
there is no verified-by-default path.

LifecycleExecutor owns one lazy-start daemon worker: control submits intents
and polls completions; the worker calls factories/services, reads (but never
mutates) the runtime holder for candidate lookup, and never touches the bus
or motion HAL. Completions are deduplicated and token-checked by the
control-side poll, deadlines bound each worker invocation (default 30 s of
supplied monotonic time), and cancellation on activation change, containment,
or shutdown drops in-flight results without spawning extra workers.

Satisfies: REQ-AIML-GIMB-002, REQ-GIMB-HIGH-001.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from flight.libs.types import ActivationKey, Err, FaultCode, Ok, Result
from flight.payload.graphs.base import (
    EffectIntent,
    EffectKind,
    EffectResult,
    EffectStatus,
    InitVerificationResult,
    InitVerificationStatus,
    TickInputs,
)
from flight.payload.graphs.parameters import GraphParameters, encoder_fresh
from flight.payload.inference.runtime import InferenceRuntime, RuntimeSession

_PREREQUISITE_KINDS: frozenset[EffectKind] = frozenset(
    {EffectKind.SELFTEST, EffectKind.MODEL_LOAD, EffectKind.HOME}
)


def _expiry_fault(kind: EffectKind) -> FaultCode:
    """Deadline fault code for one expired intent kind.

    MODEL_LOAD expiry is INFERENCE_TIMEOUT (the load/warm-up bound);
    SELFTEST, HOME, and VERIFY_INIT expiry is GIMBAL_FAULT (a blocked or
    unresponsive observation path).
    """
    if kind is EffectKind.MODEL_LOAD:
        return FaultCode.INFERENCE_TIMEOUT
    return FaultCode.GIMBAL_FAULT


@dataclass(frozen=True, slots=True)
class LifecycleToken:
    """Exact shell generation an INIT effect intent belongs to.

    Attributes:
        activation_key: Activation under which the intent was issued.
        control_revision: Control-owner revision at issue; a superseded
            revision makes pending work stale.
        containment_generation: Containment latch generation at issue; a latch
            engagement cancels the generation.
    """

    activation_key: ActivationKey
    control_revision: int
    containment_generation: int


@dataclass(frozen=True, slots=True)
class LifecycleObservation:
    """Immutable evidence snapshot handed to lifecycle services; data only.

    Attributes:
        inputs: The tick observations at the latest poll (encoder, health,
            activation key); services read observations only -- no drivers,
            bus, or clocks reach a lifecycle service.
        home_target_rad: Configured HOME pose in radians.
        issued_s: Monotonic time the intent was issued; HOME arrival requires
            an encoder sample strictly later than this.
        prerequisites: SUCCEEDED prerequisite effect results under the current
            token, in issue order.
        candidate_identity: Identity of the warm session candidate, or "".
    """

    inputs: TickInputs
    home_target_rad: float
    issued_s: float
    prerequisites: tuple[EffectResult, ...] = ()
    candidate_identity: str = ""


@runtime_checkable
class SelfTestService(Protocol):
    """Observed self-test probe for the INIT SELFTEST effect."""

    def check(
        self, observation: LifecycleObservation, cancel: threading.Event
    ) -> Result[str | None, FaultCode]:
        """Return evidence id on success, None while pending, Err on failure."""
        ...


@runtime_checkable
class HomeArrivalService(Protocol):
    """Pose-arrival probe for the INIT HOME effect."""

    def check(
        self, observation: LifecycleObservation, cancel: threading.Event
    ) -> Result[str | None, FaultCode]:
        """Return arrival evidence id, None while pending, Err on failure."""
        ...


@runtime_checkable
class InitializationVerifier(Protocol):
    """Explicit initialization-verification seam for the READY node."""

    def verify(
        self, observation: LifecycleObservation, cancel: threading.Event
    ) -> Result[InitVerificationResult, FaultCode]:
        """Return the current-key verification result, or Err on failure."""
        ...


class ObservedSelfTest:
    """Default self-test: attests only what the current observation proves.

    Scope: the encoder sample is present, finite, in-envelope, and fresh, and
    the gimbal health observation confirms feedback_valid and
    inhibit_confirmed, while containment is clear. This is not a camera or
    hardware self-test and claims nothing beyond that observation.
    """

    def __init__(self, params: GraphParameters) -> None:
        """Store the parameters supplying freshness and envelope bounds."""
        self._params = params

    def check(
        self, observation: LifecycleObservation, cancel: threading.Event
    ) -> Result[str | None, FaultCode]:
        """Succeed once observation shows fresh encoder plus confirmed inhibit."""
        if cancel.is_set():
            return Err(FaultCode.INFERENCE_TIMEOUT)
        inputs = observation.inputs
        health = inputs.health
        if health.contained or not health.feedback_valid or not health.inhibit_confirmed:
            return Ok(None)
        if not encoder_fresh(inputs, self._params):
            return Ok(None)
        encoder = inputs.encoder
        assert encoder is not None
        return Ok(f"selftest:{encoder.sample_id}")


class ExactHomeArrival:
    """Default HOME arrival: exact configured pose on fresh later feedback.

    Attests only numerical equality -- ``abs(angle - home_target_rad) <=
    1e-12`` -- on an encoder sample strictly later than the HOME intent issue
    time, with fresh valid feedback and no containment. No physical tolerance,
    dwell, or stability criterion is invented here; arrival is not a
    stable-init substitute.
    """

    def __init__(self, params: GraphParameters) -> None:
        """Store the parameters supplying freshness and envelope bounds."""
        self._params = params

    def check(
        self, observation: LifecycleObservation, cancel: threading.Event
    ) -> Result[str | None, FaultCode]:
        """Succeed on the exact home pose after the intent, else pending."""
        if cancel.is_set():
            return Err(FaultCode.INFERENCE_TIMEOUT)
        inputs = observation.inputs
        if not (
            math.isfinite(observation.home_target_rad)
            and math.isfinite(observation.issued_s)
            and math.isfinite(inputs.now_s)
        ):
            return Ok(None)
        if inputs.health.contained or not inputs.health.feedback_valid:
            return Ok(None)
        encoder = inputs.encoder
        if encoder is None or not encoder_fresh(inputs, self._params):
            return Ok(None)
        if encoder.t_s <= observation.issued_s:
            return Ok(None)
        if abs(encoder.angle_rad - observation.home_target_rad) > 1.0e-12:
            return Ok(None)
        return Ok(f"home:{encoder.sample_id}")


class PendingInitializationVerifier:
    """Production default verifier: always PENDING on the current key.

    The stable-init criterion is intentionally deferred; tests inject their
    own deterministic InitializationVerifier. There is no production
    pass-by-default option.
    """

    def verify(
        self, observation: LifecycleObservation, cancel: threading.Event
    ) -> Result[InitVerificationResult, FaultCode]:
        """Return PENDING scoped to the observation's activation key."""
        if cancel.is_set():
            return Err(FaultCode.INFERENCE_TIMEOUT)
        return Ok(
            InitVerificationResult(
                activation_key=observation.inputs.activation_key,
                status=InitVerificationStatus.PENDING,
            )
        )


@dataclass(frozen=True, slots=True)
class LifecyclePoll:
    """What one control-side executor poll produced for this tick.

    Attributes:
        results: Terminal effect results (or deadline failures) to inject into
            the graph's TickInputs.
        verification: An InitVerificationResult ready for the graph: FAILED
            results pass through graph-handled, VERIFIED only arrives together
            with ``install`` after prerequisites and evidence check out.
        install: A verified warm candidate for the control owner to install
            via ``InferenceRuntime.install_verified``, or None.
    """

    results: tuple[EffectResult, ...] = ()
    verification: InitVerificationResult | None = None
    install: RuntimeSession | None = None


@dataclass(frozen=True, slots=True)
class _PendingIntent:
    """Outstanding intent bookkeeping: the intent and its issue time."""

    intent: EffectIntent
    issued_s: float


@dataclass(slots=True)
class _Job:
    """One claimed worker execution: intent, token, cancel flag, observation."""

    intent: EffectIntent
    token: LifecycleToken
    cancel: threading.Event
    observation: LifecycleObservation


@dataclass(frozen=True, slots=True)
class _Completion:
    """One worker-produced completion for one intent under one token."""

    intent: EffectIntent
    token: LifecycleToken
    result: EffectResult | None = None
    verification: InitVerificationResult | None = None
    session: RuntimeSession | None = None


class LifecycleExecutor:
    """Single bounded worker executing INIT effect intents for the control owner.

    ``submit`` registers graph-emitted intents under a LifecycleToken; ``poll``
    (control thread only) expires overdue intents, drains a bounded completion
    mailbox, hands the next outstanding intent to the worker, and returns
    results/verification/install candidates for the app. Intents are canonical:
    at most one per EffectKind per generation, issued once and retired once, so
    a repeated graph emit can never re-execute, reinstall, or re-arm a
    completed intent. A busy worker leaves later intents pending; a blocked SDK
    call can never spawn extra workers and never blocks the control owner.
    ``cancel`` invalidates the whole generation and signals in-flight work;
    ``shutdown`` stops the worker without waiting on a blocked SDK call.

    Deadlines bound each worker invocation from its issue time (inclusive
    ``now - issued_s >= deadline``). SELFTEST, MODEL_LOAD, and HOME keep their
    original issue deadline across re-polls. VERIFY_INIT re-arms its deadline
    each time the verifier returns a current-key PENDING, so a responsive
    verifier may hold INIT indefinitely while the deferred stability criterion
    is undecided; a blocked verifier call still fails at its bound.
    """

    def __init__(
        self,
        *,
        runtime: InferenceRuntime,
        selftest: SelfTestService,
        home_arrival: HomeArrivalService,
        verifier: InitializationVerifier,
        effect_deadline_s: float = 30.0,
    ) -> None:
        """Wire services and the shared runtime holder; validate the deadline.

        Inputs:
            runtime: The app's InferenceRuntime; the MODEL_LOAD job reads its
                factory (and the installed snapshot only as the explicit
                scripted fallback). The worker never installs into it -- only
                the control owner mutates the holder.
            selftest: Probe backing the SELFTEST effect.
            home_arrival: Probe backing the HOME effect.
            verifier: Verification seam backing the VERIFY_INIT effect.
            effect_deadline_s: Finite positive bound in monotonic seconds on
                each intent execution, measured from its issue time.

        Raises:
            ValueError: If the deadline is nonfinite or nonpositive.
        """
        if not math.isfinite(effect_deadline_s) or effect_deadline_s <= 0.0:
            raise ValueError("effect_deadline_s must be finite and positive")
        self._runtime = runtime
        self._selftest = selftest
        self._home = home_arrival
        self._verifier = verifier
        self._deadline_s = effect_deadline_s
        self._condition = threading.Condition()
        self._shutdown = False
        self._thread: threading.Thread | None = None
        self._token: LifecycleToken | None = None
        self._outstanding: dict[EffectKind, _PendingIntent] = {}
        self._retired: set[EffectKind] = set()
        self._job: _Job | None = None
        self._mailbox: deque[_Completion] = deque(maxlen=4)
        self._evidence: dict[EffectKind, str] = {}
        self._candidate: RuntimeSession | None = None

    def submit(self, intents: tuple[EffectIntent, ...], token: LifecycleToken, now: float) -> None:
        """Register graph-emitted intents under the current token (control only).

        Only canonical intents register: the activation key must match the
        token and ``effect_id`` must equal ``kind.value``, giving at most four
        intents per generation. An intent already outstanding or retired under
        this generation is ignored, so a re-emitted terminal intent never
        re-executes, reinstalls, or re-arms its deadline. A token change
        without an explicit cancel still invalidates the old generation: the
        running job's cancel event is set and all generation state drops.
        """
        with self._condition:
            if self._token != token:
                self._clear_generation_locked()
                self._token = token
            for intent in intents:
                if (
                    intent.activation_key != token.activation_key
                    or intent.effect_id != intent.kind.value
                    or intent.kind in self._outstanding
                    or intent.kind in self._retired
                ):
                    continue
                self._outstanding[intent.kind] = _PendingIntent(intent=intent, issued_s=now)
            self._condition.notify_all()

    def poll(
        self, token: LifecycleToken, observation: LifecycleObservation, now: float
    ) -> LifecyclePoll:
        """Expire overdue intents, drain completions, and feed the worker.

        Runs on the control thread. Expiry is checked before consuming so a
        late completion can never promote past its deadline; an expired intent
        yields a terminal FAILED result (MODEL_LOAD -> INFERENCE_TIMEOUT, other
        kinds -> GIMBAL_FAULT) and fails the generation, cancelling all other
        pending work. Completions for another token, an unknown or already
        resolved kind, or a mismatched key/id are dropped. A terminal FAILED
        result or FAILED verification also fails the generation immediately.
        A VERIFIED verification is handed to the graph only with nonempty
        evidence, every prerequisite effect succeeded, and a warm candidate.
        """
        results: list[EffectResult] = []
        verification: InitVerificationResult | None = None
        install: RuntimeSession | None = None
        with self._condition:
            if self._token != token:
                return LifecyclePoll()
            for kind, pending in list(self._outstanding.items()):
                if now - pending.issued_s < self._deadline_s:
                    continue
                del self._outstanding[kind]
                self._retired.add(kind)
                results.append(self._failed_result(pending.intent, token, _expiry_fault(kind)))
                self._fail_generation_locked()
                break
            if not results:
                while self._mailbox:
                    completion = self._mailbox.popleft()
                    install_out, verification_out = self._consume_completion(
                        completion, token, results, now
                    )
                    if install_out is not None:
                        install = install_out
                    if verification_out is not None:
                        verification = verification_out
            if self._job is None and self._outstanding and not self._shutdown:
                pending = next(iter(self._outstanding.values()))
                self._job = _Job(
                    intent=pending.intent,
                    token=token,
                    cancel=threading.Event(),
                    observation=self._job_observation(pending, observation),
                )
                self._ensure_worker_locked()
                self._condition.notify_all()
        return LifecyclePoll(results=tuple(results), verification=verification, install=install)

    def cancel(self) -> None:
        """Invalidate the current generation; in-flight results are dropped.

        Called by the control owner on every activation change, containment
        latch, and application shutdown. The running job's own cancel event is
        set; its completion is discarded at the worker boundary and no new
        worker is spawned for it.
        """
        with self._condition:
            self._clear_generation_locked()

    def shutdown(self, join_timeout_s: float = 0.5) -> None:
        """Cancel everything and stop the worker without waiting on the SDK.

        The worker is a daemon; a blocked SDK call cannot extend shutdown. The
        join bound applies only to the worker's own exit path after the current
        call returns.
        """
        with self._condition:
            self._shutdown = True
            self._clear_generation_locked()
            self._condition.notify_all()
            thread = self._thread
        if thread is not None:
            thread.join(timeout=join_timeout_s)

    def _clear_generation_locked(self) -> None:
        """Cancel in-flight work and drop every generation record."""
        if self._job is not None:
            self._job.cancel.set()
        self._token = None
        self._outstanding.clear()
        self._retired.clear()
        self._mailbox.clear()
        self._evidence.clear()
        self._candidate = None

    def _fail_generation_locked(self) -> None:
        """End the generation on a terminal failure: no requeue, no candidate.

        Remaining outstanding intents retire without results (the graph is
        already failed on the one it receives), the running job is signalled,
        and the warm candidate is dropped so a late load can never install.
        """
        if self._job is not None:
            self._job.cancel.set()
        self._retired.update(self._outstanding)
        self._outstanding.clear()
        self._candidate = None

    def _job_observation(
        self, pending: _PendingIntent, base: LifecycleObservation
    ) -> LifecycleObservation:
        """Assemble the service observation for one outstanding intent."""
        prerequisites = tuple(
            EffectResult(
                activation_key=pending.intent.activation_key,
                effect_id=kind.value,
                kind=kind,
                status=EffectStatus.SUCCEEDED,
                evidence_id=self._evidence[kind],
            )
            for kind in EffectKind
            if kind in _PREREQUISITE_KINDS and kind in self._evidence
        )
        candidate = self._candidate
        return LifecycleObservation(
            inputs=base.inputs,
            home_target_rad=base.home_target_rad,
            issued_s=pending.issued_s,
            prerequisites=prerequisites,
            candidate_identity="" if candidate is None else candidate.identity,
        )

    def _consume_completion(
        self,
        completion: _Completion,
        token: LifecycleToken,
        results: list[EffectResult],
        now: float,
    ) -> tuple[RuntimeSession | None, InitVerificationResult | None]:
        """Classify one mailbox completion; drop anything not current.

        Only a completion matching the outstanding canonical intent (same
        kind, effect id, and activation key) can advance the generation. A
        current-key PENDING verification re-arms the intent deadline; any
        other verification shape (wrong key, blank-evidence VERIFIED, missing
        prerequisites, no candidate) neither promotes nor re-arms.
        """
        no_promotion: tuple[None, None] = (None, None)
        if completion.token != token or completion.intent.activation_key != token.activation_key:
            return no_promotion
        kind = completion.intent.kind
        pending = self._outstanding.get(kind)
        if pending is None or completion.intent.effect_id != pending.intent.effect_id:
            return no_promotion
        result = completion.result
        if result is not None:
            if result.status is EffectStatus.PENDING:
                return no_promotion
            if (
                result.activation_key != token.activation_key
                or result.kind is not kind
                or result.effect_id != pending.intent.effect_id
                or (result.status is EffectStatus.SUCCEEDED and not result.evidence_id)
            ):
                return no_promotion
            del self._outstanding[kind]
            self._retired.add(kind)
            results.append(result)
            if result.status is EffectStatus.FAILED:
                self._fail_generation_locked()
                return no_promotion
            self._evidence[kind] = result.evidence_id
            if kind is EffectKind.MODEL_LOAD and completion.session is not None:
                self._candidate = completion.session
            return no_promotion
        verification = completion.verification
        if verification is None or verification.activation_key != token.activation_key:
            return no_promotion
        if verification.status is InitVerificationStatus.PENDING:
            # A responsive verifier re-arms its invocation deadline; the
            # deferred criterion may wait indefinitely while the call itself
            # stays bounded.
            self._outstanding[kind] = _PendingIntent(pending.intent, issued_s=now)
            return no_promotion
        if verification.status is InitVerificationStatus.FAILED:
            del self._outstanding[kind]
            self._retired.add(kind)
            self._fail_generation_locked()
            return None, verification
        if not (
            verification.evidence_id
            and _PREREQUISITE_KINDS <= set(self._evidence)
            and self._candidate is not None
        ):
            # A VERIFIED response cannot promote without evidence, all
            # prerequisite effects, and a warm candidate; it stays outstanding
            # without re-arming until the verifier answers properly or the
            # deadline retires it.
            return no_promotion
        del self._outstanding[kind]
        self._retired.add(kind)
        return self._candidate, verification

    @staticmethod
    def _failed_result(
        intent: EffectIntent, token: LifecycleToken, fault: FaultCode
    ) -> EffectResult:
        """Build the terminal FAILED result for one intent under this token."""
        return EffectResult(
            activation_key=token.activation_key,
            effect_id=intent.effect_id,
            kind=intent.kind,
            status=EffectStatus.FAILED,
            fault=fault,
        )

    def _ensure_worker_locked(self) -> None:
        """Lazily start the single daemon worker on first real work."""
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._worker_loop, name="payload-lifecycle", daemon=True
            )
            self._thread.start()

    def _worker_loop(self) -> None:
        """Run claimed jobs one at a time until shutdown.

        A completion is mailed only while its generation was not cancelled and
        shutdown has not begun; a cancelled job frees the slot so pending
        work under the next generation can still run on this same worker.
        """
        while True:
            with self._condition:
                while self._job is None and not self._shutdown:
                    self._condition.wait()
                if self._shutdown:
                    return
                job = self._job
            if job is None:
                continue
            if job.cancel.is_set():
                with self._condition:
                    if self._job is job:
                        self._job = None
                continue
            completion = self._run_job(job)
            with self._condition:
                if self._job is job:
                    self._job = None
                if not job.cancel.is_set() and not self._shutdown and self._token == job.token:
                    self._mailbox.append(completion)

    def _run_job(self, job: _Job) -> _Completion:
        """Execute one intent in the worker; every failure is typed, never raised."""
        try:
            if job.intent.kind is EffectKind.MODEL_LOAD:
                return self._run_model_load(job)
            if job.intent.kind is EffectKind.SELFTEST:
                return self._run_probe(job, self._selftest)
            if job.intent.kind is EffectKind.HOME:
                return self._run_probe(job, self._home)
            return self._run_verify(job)
        except Exception:
            return _Completion(
                intent=job.intent,
                token=job.token,
                result=self._failed_result(job.intent, job.token, FaultCode.GIMBAL_FAULT),
            )

    def _run_probe(self, job: _Job, service: SelfTestService | HomeArrivalService) -> _Completion:
        """Run a selftest/home probe; Ok(id) succeeds, Ok(None) stays pending."""
        try:
            checked = service.check(job.observation, job.cancel)
        except Exception:
            return _Completion(
                intent=job.intent,
                token=job.token,
                result=self._failed_result(job.intent, job.token, FaultCode.GIMBAL_FAULT),
            )
        if isinstance(checked, Err):
            return _Completion(
                intent=job.intent,
                token=job.token,
                result=self._failed_result(job.intent, job.token, checked.error),
            )
        if not checked.value:
            return _Completion(
                intent=job.intent,
                token=job.token,
                result=EffectResult(
                    activation_key=job.intent.activation_key,
                    effect_id=job.intent.effect_id,
                    kind=job.intent.kind,
                    status=EffectStatus.PENDING,
                ),
            )
        return _Completion(
            intent=job.intent,
            token=job.token,
            result=EffectResult(
                activation_key=job.intent.activation_key,
                effect_id=job.intent.effect_id,
                kind=job.intent.kind,
                status=EffectStatus.SUCCEEDED,
                evidence_id=checked.value,
            ),
        )

    def _run_model_load(self, job: _Job) -> _Completion:
        """Load then warm up the candidate session; exceptions are MODEL_CORRUPT.

        With no configured factory the already-installed session stands in as
        the candidate (the explicit scripted composition); without either the
        load fails rather than fabricating a session.
        """
        factory = self._runtime.factory
        try:
            if factory is None:
                candidate = self._runtime.snapshot()
                if candidate is None:
                    return _Completion(
                        intent=job.intent,
                        token=job.token,
                        result=self._failed_result(job.intent, job.token, FaultCode.MODEL_CORRUPT),
                    )
            else:
                loaded = factory.load(job.cancel)
                if isinstance(loaded, Err):
                    return _Completion(
                        intent=job.intent,
                        token=job.token,
                        result=self._failed_result(job.intent, job.token, loaded.error),
                    )
                candidate = loaded.value
        except Exception:
            return _Completion(
                intent=job.intent,
                token=job.token,
                result=self._failed_result(job.intent, job.token, FaultCode.MODEL_CORRUPT),
            )
        if not candidate.identity:
            return _Completion(
                intent=job.intent,
                token=job.token,
                result=self._failed_result(job.intent, job.token, FaultCode.MODEL_CORRUPT),
            )
        if job.cancel.is_set():
            return _Completion(
                intent=job.intent,
                token=job.token,
                result=self._failed_result(job.intent, job.token, FaultCode.INFERENCE_TIMEOUT),
            )
        try:
            warmed = candidate.warm_up(job.cancel)
        except Exception:
            return _Completion(
                intent=job.intent,
                token=job.token,
                result=self._failed_result(job.intent, job.token, FaultCode.MODEL_CORRUPT),
            )
        if isinstance(warmed, Err):
            return _Completion(
                intent=job.intent,
                token=job.token,
                result=self._failed_result(job.intent, job.token, warmed.error),
            )
        return _Completion(
            intent=job.intent,
            token=job.token,
            result=EffectResult(
                activation_key=job.intent.activation_key,
                effect_id=job.intent.effect_id,
                kind=job.intent.kind,
                status=EffectStatus.SUCCEEDED,
                evidence_id=candidate.identity,
            ),
            session=candidate,
        )

    def _run_verify(self, job: _Job) -> _Completion:
        """Call the injected verifier; Err maps to a failed VERIFY_INIT result."""
        try:
            verified = self._verifier.verify(job.observation, job.cancel)
        except Exception:
            return _Completion(
                intent=job.intent,
                token=job.token,
                result=self._failed_result(job.intent, job.token, FaultCode.GIMBAL_FAULT),
            )
        if isinstance(verified, Err):
            return _Completion(
                intent=job.intent,
                token=job.token,
                result=self._failed_result(job.intent, job.token, verified.error),
            )
        return _Completion(
            intent=job.intent,
            token=job.token,
            verification=verified.value,
        )
