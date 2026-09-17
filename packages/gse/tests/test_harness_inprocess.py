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
    """Building over profiles/sil.toml then stepping yields a boot-SAFE capture."""
    backend = InProcessBackend()
    backend.build(_sil_scenario(), "profiles/sil.toml")
    for i in range(4):
        backend.step(float(i + 1))
    capture = backend.collect()
    backend.shutdown()

    # Boot is latched SAFE: no inference and no tracking motion.
    assert capture.inference_count == 0
    assert capture.mode_changes == ()
    assert capture.gimbal_moved is False


def test_socket_backend_is_deferred() -> None:
    """SocketBackend is declared but not implemented (PIL/HIL deferred)."""
    backend = SocketBackend()
    try:
        backend.build(_sil_scenario(), "profiles/pil.toml")
    except NotImplementedError as exc:
        assert "deferred" in str(exc)
    else:  # pragma: no cover - guard
        raise AssertionError("SocketBackend.build must raise NotImplementedError")
